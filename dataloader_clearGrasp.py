#!/usr/bin/env python3

'''

Dataset Class for ClearGrasp Data set.

Note: This file is adapted from ClearGrasp Dataset class implementation.

'''

import os
import glob
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
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

class ClearGraspsDataset(Dataset):
    """
    Dataset class for a subset of the ClearGrasp data set which is used for monocular depth map
    estimation. 


    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        depth_dir (str): Path to folder containing the depth maps (.exr format).
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs.
        transform (list of str): A list of imgaug Transform names to be applied only on the imgs.
        isSynthetic (bool): Whether loading synthetic data set.

    """

    def __init__(
            self,
            input_dir,
            mask_dir='',
            depth_dir='',
            mode='train',
            width=640,
            height=480,
            isSynthetic= True,
            withMask=False,
    ):

        super().__init__()

        self.images_dir = input_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.transform = iaa.Sequential([
            iaa.Resize({
                "height": height,
                "width": width
            }, interpolation='nearest'),
        ]).to_deterministic()
        self.mode = mode

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_mask = []
        self._datalist_depth = []
        if(isSynthetic):
            self._extension_input = ['-rgb.jpg']  # The file extension of input images 
            self._extension_mask = ['-segmentation-mask.png']
            self._extension_depth = ['-depth-rectified.exr']
        else:
            self._extension_input = ['-transparent-rgb-img.jpg']  # The file extension of input images  
            self._extension_mask = ['-mask.png']
            self._extension_depth = ['-opaque-depth-img.exr']
        
        self._create_lists_filenames(self.images_dir, self.mask_dir, self.depth_dir)

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
        _img = _img.resize((640, 480))
        _img = np.asarray(_img, dtype=np.float32) / 255.
        _img = np.ascontiguousarray(_img)

        # Open depths
        if self.depth_dir:
            depth_path = self._datalist_depth[index]
            _depth = exr_loader(depth_path, ndim=1)
            _depth[np.isnan(_depth)] = -1.0
            _depth[np.isinf(_depth)] = -1.0
            _depth = np.expand_dims(_depth, axis=0)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            #_img = det_tf.augment_image(_img.copy())
            if self.depth_dir:
                # Making all values of invalid pixels marked as -1.0 to 0.
                # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
                mask = np.all(_depth == -1.0, axis=0)
                _depth[:, mask] = 0.0
                _depth = _depth.transpose((1, 2, 0))  # To Shape: (H, W, 3)
                _depth = det_tf.augment_image(_depth, hooks=ia.HooksImages(activator=self._activator_masks))
                _depth = np.ascontiguousarray(_depth)

        if self.mode=='train':
            _img, _depth = self._train_preprocess(_img, _depth)

        # Return Tensors
        _img_tensor = ToTensor()(_img)

        if self.depth_dir:
            _depth_tensor = torch.from_numpy(_depth.transpose((2, 0, 1)).copy())
        else:
            _depth_tensor = torch.zeros((3, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        if self.mask_dir:
            mask_path = self._datalist_mask[index]
            _mask = Image.open(mask_path)
            _mask = _mask.resize((640, 480))
            _mask = np.asarray(_mask, dtype=np.float32)
            _mask = np.ascontiguousarray(_mask)
            _mask_tensor = torch.from_numpy(_mask)
            return _img_tensor, _mask_tensor, _depth_tensor


        return _img_tensor, _depth_tensor

    def _create_lists_filenames(self, images_dir, mask_dir, depth_dir):
        '''Creates a list of filenames of images and depths each in dataset
        The depth at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            mask_dir (str): Path to the dir where mask are stored
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

        if mask_dir:
            assert os.path.isdir(mask_dir), ('Dataloader given mask directory that does not exist: "%s"' %
                                               (mask_dir))
            for ext in self._extension_mask:
                maskSearchStr = os.path.join(mask_dir, '*' + ext)
                maskpaths = sorted(glob.glob(maskSearchStr))
                self._datalist_mask = self._datalist_mask + maskpaths

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
        return default

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def _train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self._augment_image(image)

        return image, depth_gt

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from models.DenseDepth import DenseDepth
    from models.AdaBin import UnetAdaptiveBins
    from utils import model_io
    from utils.api import depth2rgb
    import torchvision
    import torch.nn as nn

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]

    origin_path_adabins = "./models/AdaBin/weights/origin/AdaBins_nyu.pt"
    origin_path_densedepth = "./models/DenseDepth/weights/origin/nyu.h5"
    db_test = ClearGraspsDataset(input_dir='./DataSet/cleargrasp-dataset-test-val/real-val/d435',
                                    mask_dir='./DataSet/cleargrasp-dataset-test-val/real-val/d435',
                                    depth_dir='./DataSet/cleargrasp-dataset-test-val/real-val/d435',
                                    mode='eval',
                                    width=640,
                                    height=480,
                                    isSynthetic= False)

    batch_size = 4
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    device = "cuda:0"
    model = UnetAdaptiveBins.build(n_bins=256, min_val=0.1, max_val=1.5)
    #model = DenseDepth()
    if torch.cuda.device_count() > 1 and device != 'cpu':
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
    model = model.to(device)
    model = model_io.load_origin_Checkpoint(origin_path_adabins, "adabin", model)   
    model.eval()
    
    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, mask, depth = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('mask shape, type: ', mask.shape, mask.dtype)
        print('depth shape, type: ', depth.shape, depth.dtype)
        # Show Batch
        #dephs= torch.cat((imageTensor2PILTensor(img), depthTensor2rgbTensor(depth)), 2)
        #im_vis_depth = torchvision.utils.make_grid(dephs, nrow=batch_size // 2, normalize=True, scale_each=True)
#
        #plt.imshow(im_vis_depth.numpy().transpose(1, 2, 0))
        #plt.imsave("s.jpg" ,im_vis_depth.numpy().transpose(1, 2, 0), cmap='plasma_r')
        #plt.show()

        #_ , pred_adabin = model(img.to(device))
        _label = np.zeros(mask.shape, dtype=np.uint8)
        _label[mask==255] = 1
        print('mask shape, type: ', _label.shape, _label.dtype)
        _, pred_adabin = model(img.to(device))
        
        gt_masked_ = nn.functional.interpolate(depth, pred_adabin.cpu().shape[-2:], mode='bilinear', align_corners=True)
        gt_masked = gt_masked_.detach().numpy().copy()
        pred_adabin = pred_adabin.cpu().detach().numpy()
        gt_masked = gt_masked * _label
        pred_adabin = pred_adabin * _label

        print('mask shape, type: ', gt_masked.shape, mask.dtype)
        print('depth shape, type: ', pred_adabin.shape, depth.dtype)

        #for x in range(240):
        #    for y in range(320):
        #        if _label[x][y] == 0:
        #            pred_adabin[0][x][y] = 0
        #            gt_masked[0][x][y] = 0
                    

        #pred_adabin_masked = pred_adabin[0].cpu().detach().numpy() * _label
        #pred_adabin = pred_adabin[0].cpu().detach().numpy()  - pred_adabin_masked
        #print('output image shape, type: ', pred_adabin.shape, pred_adabin.dtype)
        #pred_adabin_masked = pred_adabin_masked.squeeze()
        #viz = depth2rgb(pred_adabin_masked[0], max_depth=1.5) 
        #plt.imshow(viz)
        #plt.show()
        #pred_adabin = pred_adabin.squeeze()
        #viz = depth2rgb(pred_adabin, max_depth=1.5) 
        #plt.imshow(viz)
        #plt.show()
        #gt_masked = gt_masked.squeeze()
        #viz = depth2rgb(gt_masked, max_depth=1.5) 
        #plt.imshow(viz)
        #plt.show()
        #gt_masked_ = gt_masked_[0].detach().numpy()
        #gt_masked_ = gt_masked_.squeeze()
        #viz = depth2rgb(gt_masked_, max_depth=1.5) 
        #plt.imshow(viz)
        #plt.show()
        ##plt.imshow(_label)
        #plt.imshow(img[0].cpu().detach().numpy().transpose(1,2,0))
        #plt.show()

        break
