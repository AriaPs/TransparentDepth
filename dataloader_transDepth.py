#!/usr/bin/env python3

'''

Dataset Class for TransDepth Data set.

Note: This file is adapted from TransDepth Dataset class implementation.

'''

import os
import glob
import random
from PIL import Image
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia

from utils.api import exr_loader, depthTensor2rgbTensor, imageTensor2PILTensor

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor()
    ])

class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image):
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

class TransDepthTransDepth(Dataset):
    """
    Dataset class for a subset of the TransDepth data set which is used for monocular depth map
    estimation. 


    Args:
        input_dir (str): Path to folder containing the input images (.hdf5 format).
        mode (imgaug transforms): imgaug Transforms to be applied to the imgs.
        transform (list of str): A list of imgaug Transform names to be applied only on the imgs.

    """

    def __init__(
            self,
            input_dir,
            mode='train',
    ):

        super().__init__()

        self.input_dir = input_dir
        self.transform = iaa.Sequential([
            iaa.Resize({
                "height": 512,
                "width": 512
            }, interpolation='nearest'),
        ]).to_deterministic()
        self.mode = mode

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        
        self._create_lists_filenames(self.input_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no depths directory has been specified,
        then a tensor of zeroes will be returned as the depth.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of depth (Tensor of zeroes is depth_dir is "" or None)
        '''
        
        image_path = self._datalist_input[index]
        with h5py.File(image_path) as f:
            
            # Open input imgs   
            _img = np.array(f["colors"], dtype=np.float32) / 255.
            _img = np.ascontiguousarray(_img)

            # Open depths
            _depth = np.array(f["distance"], dtype=np.float32)
            _depth[np.isnan(_depth)] = -1.0
            _depth[np.isinf(_depth)] = -1.0
            _depth = np.expand_dims(_depth, axis=0)

            # Open input masks   
            _mask = np.array(f["class_segmaps"], dtype=np.float32)
            _mask = np.expand_dims(_mask, axis=0)

            # Apply image augmentations and convert to Tensor
            if self.transform:
                det_tf = self.transform.to_deterministic()
                # Making all values of invalid pixels marked as -1.0 to 0.
                # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
                mask = np.all(_depth == -1.0, axis=0)
                _depth[:, mask] = 0.0
                _depth = _depth.transpose((1, 2, 0))  # To Shape: (H, W, 3)
                _depth = det_tf.augment_image(_depth, hooks=ia.HooksImages(activator=self._activator_masks))
                _depth = np.ascontiguousarray(_depth)

            if self.mode=='train':
                _img, _depth, _mask = self._train_preprocess(_img, _depth, _mask)

            # Return Tensors
            _img_tensor = ToTensor()(_img)

            _depth_tensor = torch.from_numpy(_depth.transpose((2, 0, 1)).copy())

            _mask_tensor = torch.from_numpy(_mask)

            return _img_tensor, _mask_tensor, _depth_tensor

    def _create_lists_filenames(self, input_dir):
        '''Creates a list of filenames of images and depths each in dataset
        The depth at index N will match the image at index N.

        Args:
            input_dir (str): Path to the dir where hdf5 files are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and depths do not match
        '''

        assert os.path.isdir(input_dir), 'Dataloader given images directory that does not exist: "%s"' % (input_dir)
        imageSearchStr = glob.glob(os.path.join(input_dir, "*", "*.hdf5"))
        imagepaths = sorted(imageSearchStr)
        self._datalist_input = self._datalist_input + imagepaths

        numImages = len(self._datalist_input)
        if numImages == 0:
            raise ValueError('No hdf5 found in given directory. Searched in dir: {} '.format(input_dir))

        

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not depths
        Eg: Blur is applied to input only, not depth. However, resize is applied to both.
        '''
        return default

    def _train_preprocess(self, image, depth_gt, mask):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            mask = (mask[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.33:
            image = self._augment_image(image)

        return image, depth_gt, mask

    def _augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

class TransDepthDataLoader(object):
    def __init__(self, input_dir, batchSize, numWorkers, mode):
        self.samples = TransDepthTransDepth(input_dir,mode=mode)
        self.data = DataLoader(self.samples, batchSize,
                                   shuffle= (mode == 'train'),
                                   num_workers=numWorkers,
                                   pin_memory=True)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    import torchvision


    db_test = TransDepthTransDepth(input_dir='/gris/gris-f/homelv/ajamili/vc/setGen/output_test/',
                                    mode='train')

    batch_size = 4
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        
    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, mask, depth = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('mask shape, type: ', mask.shape, mask.dtype)
        print('depth shape, type: ', depth.shape, depth.dtype)

        # Show Batch
        dephs= torch.cat((imageTensor2PILTensor(img), depthTensor2rgbTensor(depth)), 2)
        im_vis_depth = torchvision.utils.make_grid(dephs, nrow=batch_size // 2, normalize=True, scale_each=True)

        plt.imshow(im_vis_depth.numpy().transpose(1, 2, 0))
        plt.imsave("s.jpg" ,im_vis_depth.numpy().transpose(1, 2, 0), cmap='plasma_r')
        plt.show()
        break
