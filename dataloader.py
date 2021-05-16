#!/usr/bin/env python3

import os
import glob
import sys
from PIL import Image
import Imath
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
import cv2

from utils.utils import exr_loader, depthTensor2rgbTensor, depth2rgb


class ClearGraspsDataset(Dataset):
    """
    Dataset class for training model 

    //TODO: DOC

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs

    """

    def __init__(
            self,
            input_dir,
            depth_dir='',
            transform=None,
            input_only=None,
            outputImgWidth = 256,
            outputImgHeight = 256,
    ):

        super().__init__()

        self.images_dir = input_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_depth = []
        self._extension_input = ['-rgb.jpg']  # The file extension of input images
        self._extension_depth = ['-depth-rectified.exr']
        self._create_lists_filenames(self.images_dir, self.depth_dir)
        self.outputImgWidth = outputImgWidth
        self.outputImgHeight = outputImgHeight

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

        # Open input imgs
        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)

        # Open depths
        if self.depth_dir:
            depth_path = self._datalist_depth[index]
            _depth = exr_loader(depth_path, ndim=1) 
            #_depth = cv2.resize(_depth, (self.outputImgWidth, self.outputImgHeight), interpolation=cv2.INTER_NEAREST)
            _depth[np.isnan(_depth)] = 0
            _depth[np.isinf(_depth)] = 0
            _depth = np.expand_dims(_depth, axis=0)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img.copy())
            if self.depth_dir:
                # Making all values of invalid pixels marked as -1.0 to 0.
                # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
                mask = np.all(_depth == -1.0, axis=0)
                _depth[:, mask] = 0.0

                _depth = _depth.transpose((1, 2, 0))  # To Shape: (H, W, 3)
                _depth = det_tf.augment_image(_depth, hooks=ia.HooksImages(activator=self._activator_masks))
                _depth = _depth.transpose((2, 0, 1))  # To Shape: (3, H, W)

            
        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img.copy())

        if self.depth_dir:
            _depth_tensor = torch.from_numpy(_depth.copy())
            #_depth_tensor = nn.functional.normalize(_depth_tensor, p=2, dim=0)
        else:
            _depth_tensor = torch.zeros((3, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _depth_tensor

    def _create_lists_filenames(self, images_dir, depth_dir):
        '''Creates a list of filenames of images and depths each in dataset
        The depth at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            depth_dir (str): Path to the dir where depths are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and depths do not match
        '''

        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        for ext in self._extension_input:
            imageSearchStr = os.path.join(images_dir, '*' + ext)
            imagepaths = sorted(glob.glob(imageSearchStr))
            self._datalist_input = self._datalist_input + imagepaths

        numImages = len(self._datalist_input)
        if numImages == 0:
            raise ValueError('No images found in given directory. Searched in dir: {} '.format(images_dir))

        if depth_dir:
            assert os.path.isdir(depth_dir), ('Dataloader given depths directory that does not exist: "%s"' %
                                               (depth_dir))
            for ext in self._extension_depth:
                depthSearchStr = os.path.join(depth_dir, '*' + ext)
                depthpaths = sorted(glob.glob(depthSearchStr))
                self._datalist_depth = self._datalist_depth + depthpaths

            numdepths = len(self._datalist_depth)
            if numdepths == 0:
                raise ValueError('No depths found in given directory. Searched for {}'.format(imageSearchStr))
            if numImages != numdepths:
                raise ValueError('The number of images and depths do not match. Please check data,' +
                                 'found {} images and {} depths in dirs:\n'.format(numImages, numdepths) +
                                 'images: {}\ndepths: {}\n'.format(images_dir, depth_dir))
        

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not depths
        Eg: Blur is applied to input only, not depth. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision
    import imageio

    # Example Augmentations using imgaug
    imsize = 512
    augs_train = iaa.Sequential([
        # Geometric Augs
         iaa.Scale((imsize, imsize), 0), # Resize image
         iaa.Fliplr(0.5),
         iaa.Flipud(0.5),
         iaa.Rot90((0, 4)),
         # Blur and Noise
         iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
         iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
         iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
         # Color, Contrast, etc.
         iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
         iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
         iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
         iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
     ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])
    min = 0.1
    max = 1.5
    augs = augs_train
    input_only = ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]
    l = '/mnt/hgfs/Se6/c/test/BT/DenseDepth/data/train/depth-imgs-rectified/d.png'
    db_test = ClearGraspsDataset(input_dir='/mnt/hgfs/Se6/c/test/BT/DenseDepth/data/train/rgb-imgs',
                                    depth_dir='/mnt/hgfs/Se6/c/test/BT/DenseDepth/data/train/depth-imgs-rectified',
                                    transform=augs,
                                    input_only=input_only)

    batch_size = 4
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    
    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, depth = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('depth shape, type: ', depth.shape, depth.dtype)
        # Show Batch
        im_vis1 = torchvision.utils.make_grid(img, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis1.numpy().transpose(1, 2, 0))
        plt.show()
        im_vis2 = torchvision.utils.make_grid(depthTensor2rgbTensor(depth), nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis2.numpy().transpose(1, 2, 0))
        plt.show()

        break
