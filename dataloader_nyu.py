# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class NYUDepthDataLoader(object):
    def __init__(self, config, mode, useImgaug=False):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(config, mode, transform=preprocessing_transforms(mode),useImgaug=useImgaug)
            self.data = DataLoader(self.training_samples, config.eval.batchSize,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.eval.numWorkers,
                                   pin_memory=True)
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(config, mode, useImgaug)
            self.data = DataLoader(self.testing_samples, config.eval.batchSize,
                                   shuffle= False,
                                   num_workers=config.eval.numWorkers,
                                   pin_memory=True)

        else:
            print('mode should be one of \'train, test\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, useImgaug=False):
        self.config = config
        
        with open(config.eval.otherSets.filenames_file_nyu, 'r') as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        if useImgaug:
            self.to_tensor = ToTensor_Imgaug
        else:
            self.to_tensor = ToTensor

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])

        data_path = self.config.eval.datasetNYU.images
        image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]))
        image = Image.open(image_path).convert('RGB')
        gt_path = self.config.eval.datasetNYU.depths
        depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
        has_valid_depth = False
        try:
            depth_gt = Image.open(depth_path)
            has_valid_depth = True
        except IOError:
            depth_gt = False
            # print('Missing gt for {}'.format(image_path))
        if has_valid_depth:
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 1000.0
        
        #sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
        #              'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}

        #if self.transform:
        #    sample = self.transform(sample)



        return ToTensor()(image, depth_gt)

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.eval.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)

class ToTensor_Imgaug(object):
    def __call__(self, image, depth):
        transform = iaa.Sequential([
            iaa.Resize({
                "height": 480,
                "width": 640
            }, interpolation='nearest'),
        ]).to_deterministic()
        _img = np.asarray(image)
        _img = np.ascontiguousarray(_img)
        _img = transform.augment_image(_img)
        _img_tensor = transforms.ToTensor()(_img) / 255
        _depth_tensor = torch.from_numpy(depth.transpose((2, 0, 1)))  # To Shape: (H, W, 3)
        return _img_tensor, _depth_tensor


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, depth, target_size=(640, 480)):
        image = image.resize(target_size)
        image = np.asarray(image, dtype=np.float32) / 255.
        image = np.ascontiguousarray(image)
        image = self.to_tensor(image)
        image = self.normalize(image)
        _depth_tensor = torch.from_numpy(depth.transpose((2, 0, 1)))  # To Shape: (H, W, 3)
        return image, _depth_tensor

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision
    from attrdict import AttrDict

    from utils.api import depthTensor2rgbTensor, imageTensor2PILTensor


    config_yaml = {
        'eval': {
            'batchSize': 4,
            'numWorkers': 4,
            'otherSets': {'filenames_file_nyu': './DataSet/nyu_depth_v2/nyudepthv2_test_files_with_gt.txt'
                        },
            'datasetNYU': {
                            'images': './DataSet/nyu_depth_v2/official_splits/test/',
                            'depths': './DataSet/nyu_depth_v2/official_splits/test/'
                          },
            'dataset': 'nyu'
        }
    }
    
    config = AttrDict(config_yaml)
    

    batch_size = 4
    testloader = NYUDepthDataLoader(config, 'test').data
    
    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, depth = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('depth shape, type: ', depth.shape, depth.dtype)
        # Show Batch
        dephs= torch.cat((imageTensor2PILTensor(img), depthTensor2rgbTensor(depth)), 2)
        im_vis_depth = torchvision.utils.make_grid(dephs, nrow=batch_size // 2, normalize=True, scale_each=True)
        plt.imshow(im_vis_depth.numpy().transpose(1, 2, 0))
        plt.show()

        break