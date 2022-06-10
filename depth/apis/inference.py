# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from PIL import Image
import h5py
from depth.datasets.pipelines import Compose
from depth.models import build_depther


def init_depther(config, checkpoint=None, device='cuda:5'):
    """Initialize a depther from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed depther.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_depther(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        #if isinstance(results['img'], str):
        #    results['filename'] = results['img']
        #    results['ori_filename'] = results['img']
        #else:
        #    results['filename'] = None
        #    results['ori_filename'] = None
        # results['pad_shape'] = None
        #results['scale_factor'] = None
        #results['flip'] = None
        #results['flip_direction'] = None
        #results['img_norm_cfg'] = None
        #results['cam_intrinsic'] = None
        with h5py.File(results) as f:
            # Open input imgs   
            img = np.array(f["colors"], dtype=np.float32) / 255.
            #results['img'] = img
            #results['img_shape'] = img.shape
            #results['ori_shape'] = img.shape
            return img


def inference_depther(model, img):
    """Inference image(s) with the depther.

    Args:
        model (nn.Module): The loaded depther.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The depth estimation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=LoadImage()(img), img_metas=None)
    #data = test_pipeline(data)
    #data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    #print(data['img'])
    print(data)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


if __name__ == '__main__':
    import numpy as np
    model = init_depther("/local/ajamili/models/Toolbox/configs/depthformer/depthformer_swinl_22k_w7_nyu.py", checkpoint="/local/ajamili/models/Toolbox/configs/depthformer/weights/origin/depthformer_swinl_22k_nyu.pth", device='cuda:5')
    #im = inference_depther(model, "/local/ajamili/test_img/img/000000005-rgb.jpg") 
    im = inference_depther(model, "/gris/gris-f/homelv/ajamili/DataSet/transDepth/train/0/0.hdf5")
    print(np.shape(im))
