'''
This module contains the loss functions used to train the models.

Note: This file is adapted from AdaBins and DenseDepth framework.

'''

import torch
import torch.nn as nn
import kornia
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence



"""
Taken from Python implematation of AdaBins 
--> Computes SI loss for the model output and ground truth provided as tensors. 
"""
class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        safe_log = lambda x: torch.log(torch.clamp(x, min=1e-6))

        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        mask_invalid_pixels = torch.all(target <= 0)
        target[mask_invalid_pixels] = 1e-6
        input[mask_invalid_pixels] = 1e-6


        g = safe_log(input) - safe_log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

"""
Taken from Python implematation of AdaBins 
--> Computes chamfer loss for the AdaBins module output and ground truth provided as tensors. 
"""
class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"
    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
       # n, c, h, w = target_depth_maps.shape
        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1
        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss


"""
Adapted from Python implematation of DenseDepth 
--> Computes SSIM loss for the model output and ground truth provided as tensors. It upsamples the model output
 to ground truth size if [interpolate] is true

"""
def ssim(model_output, gt_vec, interpolate=True):
    if interpolate:
        model_output = nn.functional.interpolate(model_output, gt_vec.shape[-2:], mode='bilinear', align_corners=True)
    # calculate loss only on valid pixels
    mask_invalid_pixels = torch.all(gt_vec <= 0)
    gt_vec[mask_invalid_pixels] = 0.0
    model_output[mask_invalid_pixels] = 0.0
    criterion = nn.L1Loss()
    l_depth = criterion(model_output, gt_vec)
    ssim = kornia.losses.SSIM(window_size=11,max_val=1.5/0.1)
    l_ssim = torch.clamp((1 - ssim(model_output, gt_vec)) * 0.5, 0, 1)
    loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)
    return loss



def compute_errors(gt, pred, should_masked=True, dataset="clearGrasp", masks=None,):
    '''Returns a dictionary containig the result of all evaluation metric.

        Args:
            gt (Torch.tensor): The ground truth
            pred (Torch.tensor): The network output
            should_masked (bool): Whether mask all invalid pixel

        Returns:
            dict(str, float): A dictionary having the name of metric as key and the result as value
    '''

    safe_log = lambda x: np.log(np.clip(x, 1e-6, 1e6))
    safe_log10 = lambda x: np.log10(np.clip(x, 1e-6, 1e6))

    if masks !=None :
        if dataset == "nyu":
            mask = (gt <= 0)
            mask[45:471, 41:601] = 0
        else:
            mask = (masks == 0)
            #_label = np.zeros(masks.shape, dtype=np.uint8)
            #_label[masks==1] = 1
            #print(_label.shape, pred.shape, gt.shape)
            #gt = gt * _label
            #pred = pred * _label
            #gt = gt.flatten()
            #pred = pred.flatten()
            #_label = _label.flatten()
            ##print(_label.shape, pred.shape, gt.shape)
            #m = _label == 1
            #gt = gt[m]
            #pred = pred[m]
        
        gt = np.ma.masked_array(gt, mask=mask)
        pred = np.ma.masked_array(pred, mask=mask)
    
    else:
        gt = gt.numpy()
        pred = pred.numpy()
        
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    err = gt - pred
    abs_rel = np.mean(np.abs(err) / gt)
    sq_rel = np.mean((err ** 2) / gt)

    rmse = np.sqrt(np.mean(err ** 2))

    rmse_log = (safe_log(gt) - safe_log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = safe_log(pred) - safe_log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(safe_log10(gt) - safe_log10(pred))).mean()

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)
