
'''

Inference the trained models for depth map estimation of transparent structures

Note: This file is adapted from AdaBins inference framework.

'''

import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from imgaug import augmenters as iaa
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from torchinfo import summary
import matplotlib.pyplot as plt


from utils import model_io
from utils.api import depth2rgb, exr_saver
from models.DenseDepth import DenseDepth
from models.AdaBin import UnetAdaptiveBins
from models.DPT.models import DPTDepthModel
from models.LapDepth import LDRN


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]


pretrained_path_adbins = "./models/AdaBin/weights/trained/adabin_SSIM.pth"
pretrained_path_densedepth = "./models/DenseDepth/weights/trained/densedepth_SSIM.pth"
pretrained_path_dpt = "./models/DPT/weights/trained/dpt_SSIM_imgaug.pth"
pretrained_path_lapdepth = "./models/LapDepth/weights/trained/lapdepth_SSIM.pth"

origin_path_adabins = "./models/AdaBin/weights/origin/AdaBins_nyu.pt"
origin_path_densedepth = "./models/DenseDepth/weights/origin/nyu.h5"
origin_path_dpt = "./models/DPT/weights/origin/dpt_hybrid_nyu-2ce69ec7.pt"
origin_path_lapdepth = "./models/LapDepth/weights/origin/LDRN_NYU_ResNext101_pretrained_data.pkl"

class ToTensor_ClearGrasp(object):
    def __call__(self, image, transform):
        _img = np.asarray(image)
        _img = transform.augment_image(_img)
        _img_tensor = transforms.ToTensor()(_img) /255.0
        return _img_tensor

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class ToTensor_NYU(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480), model='adabin'):
        image = image.resize(target_size)
        image = np.asarray(image, dtype=np.float32) / 255.
        image = np.ascontiguousarray(image)
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


class InferenceHelper:
    """
    InferenceHelper class provides function for generate depth maps with lerned weights.

    Args:
        model (str): The model name which should be used. If None, all trained models a grid image with input and output
                    of all models is going to be generated for each input image.
        device (str): The device on which the inference should be applied. ('cpu' or 'gpu:id')

    """

    def __init__(self, model=None, device='cuda:0', dataset='clearGrasp', dataloaderUsage=[False,False,False,False]):
        self.device = device
        self.min_depth = 0.1
        self.dataset = dataset
        self.max_depth = 10
        self.model_name = model
        self.dataloaderUsage = dataloaderUsage
        self.transform = iaa.Sequential([
            iaa.Resize({
                "height": 480,
                "width": 640
            }, interpolation='nearest'),
        ]).to_deterministic()


        if model == 'adabin':
            self.model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            if dataset== 'clearGrasp':
                pretrained_path = pretrained_path_adbins
            else:
                pretrained_path = origin_path_adabins
        elif model == 'densedepth':
            self.model = DenseDepth()
            if dataset== 'clearGrasp':
                pretrained_path = pretrained_path_densedepth
            else:
                pretrained_path = origin_path_densedepth
        elif model == 'dpt':
            self.model = DPTDepthModel(
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            ) 
            if dataset== 'clearGrasp':
                pretrained_path = pretrained_path_dpt
            else:
                pretrained_path = origin_path_dpt
        elif model == 'lapdepth':
            self.model = LDRN({
                'lv6': False,
                'encoder': 'ResNext101',
                'norm': 'BN',
                'act': 'ReLU',
                'max_depth': self.max_depth,
            })
            if dataset== 'clearGrasp':
                pretrained_path = pretrained_path_lapdepth
            else:
                pretrained_path = origin_path_lapdepth
        elif model is None:
            self.model_adabins = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            self.model_densedepth = DenseDepth()
            self.model_dpt = DPTDepthModel(
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.model_lapdepth = LDRN({
                'lv6': False,
                'encoder': 'ResNext101',
                'norm': 'BN',
                'act': 'ReLu',
                'max_depth': self.max_depth,
            })
        else:
            raise ValueError("model can be either None, 'adabin', 'densedepth' or 'dpt' but got {}".format(model))

        if model is not None:
            
            if torch.cuda.device_count() > 1 and device != 'cpu':
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
            if dataset== 'clearGrasp':
                _ , self.model = model_io.load_checkpoint(pretrained_path, self.model)
            else:
                self.model = model_io.load_origin_Checkpoint(pretrained_path, model, self.model)

            self.model.eval()
        else:
            if dataset== 'clearGrasp':
                _ , self.model_adabins = model_io.load_checkpoint(pretrained_path_adbins, self.model_adabins)
                _ , self.model_densedepth = model_io.load_checkpoint(pretrained_path_densedepth, self.model_densedepth)
                _ , self.model_dpt = model_io.load_checkpoint(pretrained_path_dpt, self.model_dpt)
                _ , self.model_lapdepth = model_io.load_checkpoint(pretrained_path_lapdepth, self.model_lapdepth)
            else:
                self.model_adabins = model_io.load_origin_Checkpoint(origin_path_adabins, 'adabin', self.model_adabins)
                self.model_densedepth = model_io.load_origin_Checkpoint(origin_path_densedepth, 'densedepth', self.model_densedepth)
                self.model_dpt = model_io.load_origin_Checkpoint(origin_path_dpt, 'dpt', self.model_dpt)
                self.model_lapdepth = model_io.load_origin_Checkpoint(origin_path_lapdepth, 'lapdepth', self.model_lapdepth)

            if torch.cuda.device_count() > 1 and device != 'cpu':
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model_adabins = nn.DataParallel(self.model_adabins)
                self.model_densedepth = nn.DataParallel(self.model_densedepth)
                self.model_dpt = nn.DataParallel(self.model_dpt)
                self.model_lapdepth = nn.DataParallel(self.model_lapdepth)
            
            self.model_adabins = self.model_adabins.to(self.device)
            self.model_lapdepth = self.model_lapdepth.to(self.device) 
            self.model_densedepth = self.model_densedepth.to(self.device)
            self.model_dpt = self.model_dpt.to(self.device)
            
            self.model_adabins.eval()
            self.model_densedepth.eval()
            self.model_dpt.eval()
            self.model_lapdepth.eval()
            
                

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        '''Returns the output of the inference on [pil_image] by the specified model as a numpy.ndarray. Moreover, it returns also a visualization 
        of the depth map or a grid image if [visualized] is true. 

        Args:
            pil_image (PIL.Image): The input as Image class 
            visualized (bool): Whether to generate a visualization of the depth maps

        Returns:
            numpy.ndarray: The output of models
            numpy.ndarray: The visualization of the depth map or grid image
        '''
        
        _img_tensor_imaug = ToTensor_ClearGrasp()(pil_image, transform= self.transform).unsqueeze(0).to(self.device)
        _img_tensor_np = ToTensor_NYU()(pil_image).unsqueeze(0).float().to(self.device)

        if self.model_name is None:
            pred_adabin, pred_densedepth, pred_dpt, pred_lapdepth = self.predict(_img_tensor_imaug, _img_tensor_np) 
            if visualized:
                size = (320,240)
                output_rgb_adabin = depth2rgb(pred_adabin)
                output_rgb_adabin = cv2.resize(output_rgb_adabin, size, interpolation=cv2.INTER_LINEAR)
                output_rgb_densedepth = depth2rgb(pred_densedepth)
                output_rgb_densedepth = cv2.resize(output_rgb_densedepth, size, interpolation=cv2.INTER_LINEAR)
                output_rgb_dpt = depth2rgb(pred_dpt)
                output_rgb_dpt = cv2.resize(output_rgb_dpt, size, interpolation=cv2.INTER_LINEAR)
                output_rgb_lapdepth = depth2rgb(pred_lapdepth)
                output_rgb_lapdepth = cv2.resize(output_rgb_lapdepth, size, interpolation=cv2.INTER_LINEAR)

                img = cv2.normalize(_img_tensor_imaug[0].cpu().numpy().transpose(1, 2, 0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
                viz = np.concatenate((img, output_rgb_densedepth, output_rgb_lapdepth, output_rgb_adabin, output_rgb_dpt), 1)
                return pred_adabin, pred_densedepth, pred_dpt, pred_lapdepth, viz
            
            return pred_adabin, pred_densedepth, pred_dpt, pred_lapdepth
        else:
            pred = self.predict(_img_tensor_np) 
            if visualized:
                viz = depth2rgb(pred, min_depth=self.min_depth, max_depth=self.max_depth)
                return pred, viz
        
            return pred

    @torch.no_grad()
    def predict(self, img_imaug, image_np):
        '''Returns the output of the inference on [image] by the specified model as a numpy.ndarray.

        Args:
            pil_image (PIL.Image): The input as Image class

        Returns:
            numpy.ndarray: The output of models
        '''
        if self.model_name is None:
            _ , pred_adabin = self.model_adabins(img_imaug if self.dataloaderUsage[2] else image_np) 
            _ , pred_lapdepth = self.model_lapdepth(img_imaug if self.dataloaderUsage[1] else image_np)
            pred_densedepth = self.model_densedepth(img_imaug if self.dataloaderUsage[0] else image_np) 
            pred_dpt = self.model_dpt(img_imaug if self.dataloaderUsage[3] else image_np)
            
            pred_adabin = pred_adabin.cpu().detach().numpy().squeeze()
            pred_densedepth = pred_densedepth.cpu().detach().numpy().squeeze()

            pred_densedepth = np.clip(self.max_depth/pred_densedepth, self.min_depth, self.max_depth) 

            pred_dpt = pred_dpt.cpu().detach().numpy().squeeze()
            pred_lapdepth = pred_lapdepth.cpu().detach().numpy().squeeze()
            return pred_adabin, pred_densedepth, pred_dpt, pred_lapdepth
        else:
            if self.model_name == 'adabin' or self.model_name == 'lapdepth':
                _ , pred = self.model(image_np)
            else:
                pred = self.model(image_np)
                                
            pred = pred.cpu().numpy().squeeze()
            
            return pred

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir, save_exr=False):
        '''Generates for each image files in [test_dir] the depth map visualization
        and if [save_exr] is true, also an exr file. The generated files are saved in [out_dir].
        Note: [test_dir] should contain only image files

        Args:
            test_dir (str): The path of input img
            out_dir (str): The path where the output should be saved
            save_exr (bool): Wheter save the depth maps as exr files

        '''
        os.makedirs(out_dir, exist_ok=True)
        
        all_files = glob.glob(os.path.join(test_dir, "*"))

        for f in tqdm(all_files):
            if os.path.isdir(f):
                continue

            img = Image.open(f).convert('RGB')
            
            if self.model_name is None:
                pred_adabin, pred_densedepth, pred_dpt, pred_lapdepth, pred_viz = self.predict_pil(img, visualized= True)
            else:
                pred, pred_viz = self.predict_pil(img, visualized= True)
        
            basename = os.path.basename(f).split('.')[0]
            save_path_img = os.path.join(out_dir, basename + "-depth.png")

            if save_exr:
                if self.model_name is None:
                    save_path_exr_adabin = os.path.join(out_dir, basename + "-adabin.exr")
                    save_path_exr_densedepth = os.path.join(out_dir, basename + "-densedepth.exr")
                    save_path_exr_dpt = os.path.join(out_dir, basename + "-dpt.exr")
                    save_path_exr_lapdepth = os.path.join(out_dir, basename + "-lapdepth.exr")
                    exr_saver(save_path_exr_adabin, pred_adabin)
                    exr_saver(save_path_exr_densedepth, pred_densedepth)
                    exr_saver(save_path_exr_dpt, pred_dpt)
                    exr_saver(save_path_exr_lapdepth, pred_lapdepth)
                else:
                    save_path_exr = os.path.join(out_dir, basename + ".exr")
                    exr_saver(save_path_exr, pred)
            Image.fromarray(pred_viz).save(save_path_img)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Supported Models: adabins, densedepth and dpt
    inferHelper = InferenceHelper(dataloaderUsage=[False,False,False,False], dataset='clearGrasp')

    # That is how you predict depth from images in a path
    inferHelper.predict_dir("test_img/img", "test_img/ress/", save_exr=False)

    # That is how you predict a pil image
    #img = Image.open("test_img/img/000000002-transparent-rgb-img.jpg").convert('RGB')
    #_, _, _, _, viz = inferHelper.predict_pil(pil_image=img, visualized=True)
    #plt.imshow(viz)
    #plt.savefig("test_img/res.png")