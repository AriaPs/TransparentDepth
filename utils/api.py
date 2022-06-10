'''
Misc functions like functions for reading and saving EXR images using OpenEXR, saving pointclouds, etc. 
adapted or taken from ClearGarsp framework.

'''
import struct

import numpy as np
import cv2
import Imath
import OpenEXR
from PIL import Image
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array

    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)

    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


def exr_saver(EXR_PATH, ndarr, ndim=1):
    '''Saves a numpy array as an EXR file with HALF precision (float16)
    Args:
        EXR_PATH (str): The path to which file will be saved
        ndarr (ndarray): A numpy array containing img data
        ndim (int): The num of dimensions in the saved exr image, either 3 or 1.
                        If ndim = 3, ndarr should be of shape (height, width) or (3 x height x width),
                        If ndim = 1, ndarr should be of shape (height, width)
    Returns:
        None
    '''
    if ndim == 3:
        # Check params
        if len(ndarr.shape) == 2:
            # If a depth image of shape (height x width) is passed, convert into shape (3 x height x width)
            ndarr = np.stack((ndarr, ndarr, ndarr), axis=0)

        if ndarr.shape[0] != 3 or len(ndarr.shape) != 3:
            raise ValueError(
                'The shape of the tensor should be (3 x height x width) for ndim = 3. Given shape is {}'.format(
                    ndarr.shape))

        # Convert each channel to strings
        Rs = ndarr[0, :, :].astype(np.float16).tostring()
        Gs = ndarr[1, :, :].astype(np.float16).tostring()
        Bs = ndarr[2, :, :].astype(np.float16).tostring()

        # Write the three color channels to the output file
        HEADER = OpenEXR.Header(ndarr.shape[2], ndarr.shape[1])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs, 'G': Gs, 'B': Bs})
        out.close()
    elif ndim == 1:
        # Check params
        if len(ndarr.shape) != 2:
            raise ValueError(('The shape of the tensor should be (height x width) for ndim = 1. ' +
                              'Given shape is {}'.format(ndarr.shape)))

        # Convert each channel to strings
        Rs = ndarr[:, :].astype(np.float16).tostring()

        # Write the color channel to the output file
        HEADER = OpenEXR.Header(ndarr.shape[1], ndarr.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "R"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs})
        out.close()


def save_uint16_png(path, image):
    '''save weight file - scaled png representation of outlines estimation

        Args:
            path (str): path to save the file
            image (numpy.ndarray): 16-bit single channel image to be saved.
                                          Shape=(H, W), dtype=np.uint16
        '''
    assert image.dtype == np.uint16, ("data type of the array should be np.uint16." + "Got {}".format(image.dtype))
    assert len(image.shape) == 2, ("Shape of input image should be (H, W)" + "Got {}".format(len(image.shape)))

    array_buffer = image.tobytes()
    img = Image.new("I", image.T.shape)
    img.frombytes(array_buffer, 'raw', 'I;16')
    img.save(path)


def _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=0.0, max_depth=2.5):
    '''Converts a floating point depth image to uint8 or uint16 image.
    The depth image is first scaled to (0.0, max_depth) and then scaled and converted to given datatype.

    Args:
        depth_img (numpy.float32): Depth image, value is depth in meters
        dtype (numpy.dtype, optional): Defaults to np.uint16. Output data type. Must be np.uint8 or np.uint16
        max_depth (float, optional): The max depth to be considered in the input depth image. The min depth is
            considered to be 0.0.
    Raises:
        ValueError: If wrong dtype is given

    Returns:
        numpy.ndarray: Depth image scaled to given dtype
    '''

    if dtype != np.uint16 and dtype != np.uint8:
        raise ValueError('Unsupported dtype {}. Must be one of ("np.uint8", "np.uint16")'.format(dtype))

    # Clip depth image to given range
    depth_img = np.ma.masked_array(depth_img, mask=(depth_img == 0.0))
    depth_img = np.ma.clip(depth_img, min_depth, max_depth)

    # Get min/max value of given datatype
    type_info = np.iinfo(dtype)
    min_val = type_info.min
    max_val = type_info.max

    # Scale the depth image to given datatype range
    depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
    depth_img = depth_img.astype(dtype)

    depth_img = np.ma.filled(depth_img, fill_value=0)  # Convert back to normal numpy array from masked numpy array

    return depth_img


def depth2rgb(depth_img, min_depth=0.0, max_depth=2.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False,
              dynamic_scaling=False):
    '''Generates RGB representation of a depth image.
    To do so, the depth image has to be normalized by specifying a min and max depth to be considered.

    Holes in the depth image (0.0) appear black in color.

    Args:
        depth_img (numpy.ndarray): Depth image, values in meters. Shape=(H, W), dtype=np.float32
        min_depth (float): Min depth to be considered
        max_depth (float): Max depth to be considered
        color_mode (int): Integer or cv2 object representing Which coloring scheme to use.
                          Please consult https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html

                          Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
                          This mapping changes from version to version.
        reverse_scale (bool): Whether to make the largest values the smallest to reverse the color mapping
        dynamic_scaling (bool): If true, the depth image will be colored according to the min/max depth value within the
                                image, rather that the passed arguments.
    Returns:
        numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)
    '''

    # Map depth image to Color Map
    if dynamic_scaling:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8,
                                                min_depth=max(depth_img[depth_img > 0].min().item(), min_depth) ,    # Add a small epsilon so that min depth does not show up as black (invalid pixels)
                                                max_depth=min(depth_img.max().item() , max_depth))
    else:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)

    if reverse_scale is True:
        depth_img_scaled = np.ma.masked_array(depth_img_scaled, mask=(depth_img_scaled == 0.0))
        depth_img_scaled = 255 - depth_img_scaled
        depth_img_scaled = np.ma.filled(depth_img_scaled, fill_value=0)

    depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
    depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

    # Make holes in input depth black:
    depth_img_mapped[depth_img_scaled == 0, :] = 0

    return depth_img_mapped


def depthTensor2rgbTensor(depth_img, min_depth=0.0, max_depth=2.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False,
              dynamic_scaling=False):
    '''Generates RGB representation of each depth images from a batch tensor.
    To do so, the depth image has to be normalized by specifying a min and max depth to be considered.

    Holes in the depth image (0.0) appear black in color.

    Args:
        depth_img (Torch.Tensor): Depth images in a batch tensor
        min_depth (float): Min depth to be considered
        max_depth (float): Max depth to be considered
        color_mode (int): Integer or cv2 object representing Which coloring scheme to use.
                          Please consult https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html

                          Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
                          This mapping changes from version to version.
        reverse_scale (bool): Whether to make the largest values the smallest to reverse the color mapping
        dynamic_scaling (bool): If true, the depth image will be colored according to the min/max depth value within the
                                image, rather that the passed arguments.
    Returns:
        Torch.Tensor: A batch of RGB representation of depth images. Shape=(B,3,H,W)
    '''
    res = np.zeros(shape= (depth_img.shape[0], 3, depth_img.shape[2],depth_img.shape[3]))
    
    for i in range(depth_img.shape[0]):
        img = depth_img[i][0]
        # Map depth image to Color Map
        if dynamic_scaling:
            depth_img_scaled = _normalize_depth_img(img, dtype=np.uint8,
                                                    min_depth=max(depth_img[depth_img > 0].min().item() , min_depth),    # Add a small epsilon so that min depth does not show up as black (invalid pixels)
                                                    max_depth=min(depth_img.max().item() , max_depth))
        else:
            depth_img_scaled = _normalize_depth_img(img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)

        if reverse_scale is True:
            depth_img_scaled = np.ma.masked_array(depth_img_scaled, mask=(depth_img_scaled == 0.0))
            depth_img_scaled = 255 - depth_img_scaled
            depth_img_scaled = np.ma.filled(depth_img_scaled, fill_value=0)

        depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
        depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

        # Make holes in input depth black:
        depth_img_mapped[depth_img_scaled == 0, :] = 0
        depth_img_mapped = depth_img_mapped.transpose(2,0,1) # transpose it to (3, H , W)
        res[i] = depth_img_mapped.copy()

    return torch.from_numpy(res)

def imageTensor2PILTensor(img_tensor):
    '''Converts each image in the batch of images to normalized RGB visualization

    Args:
        depth_img_tensor (Torch.Tensor): A batch of images
    Returns:
        Torch.Tensor: The converted RGB representation. Shape=(B,3,H,W)
    '''
    res = np.zeros(shape= (img_tensor.shape[0], 3, img_tensor.shape[2], img_tensor.shape[3]))
    
    for i in range(img_tensor.shape[0]):
        img = img_tensor[i]
        I = cv2.normalize(img.numpy().transpose(1, 2, 0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img_mapped = np.array(I).transpose(2, 0, 1)
        res[i] = img_mapped

    return torch.from_numpy(res)

def create_grid_image(inputs, outputs, depth):
    '''Make a grid of images for display purposes.
    Size of grid is (B, 3), where each coloum belongs to input, output, ground truth

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        depth (Tensor): Batch Tensor of shape (B x C x H x W)

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''
    batch_size = inputs.shape[0]
    input_rgb = imageTensor2PILTensor(inputs)
    output_rgb = depthTensor2rgbTensor(outputs)
    depth_rgb = depthTensor2rgbTensor(depth)

    dephs= torch.cat((input_rgb, depth_rgb, output_rgb), 2)
    im_vis_depth = make_grid(dephs, nrow=batch_size // 2, normalize=True, scale_each=True)
        
    return im_vis_depth
