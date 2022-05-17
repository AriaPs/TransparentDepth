
import os

import numpy as np
import torch
from termcolor import colored
import tensorflow as tf


def save_model(CHECKPOINT_DIR, epoch, model, optimizer, total_iter_num, epoch_loss, config_yaml, name):
    '''Saves the state_dict of [model] and [optimizer], [total_iter_num], [epoch_loss], and [config_yaml] to [CHECKPOINT_DIR]. 
    The checkpoint will have the file name: [name].pth

        Args:
            CHECKPOINT_DIR (str): The save path of checkpoint file 
            epoch (int): Number of trained epochs
            model (torch.nn.Module): The model which was trained 
            optimizer (torch.optim.AdamW): Instance of optimizer
            total_iter_num (int): Total number of iteration seen by model
            epoch_loss (float): The epoch loss
            config_yaml (dict(str,str)): Configuration yaml file
            name (str): The checkpoint file name

    '''

    filename = os.path.join(CHECKPOINT_DIR, name)   
    if torch.cuda.device_count() > 1:
        model_params = model.module.state_dict()  # Saving nn.DataParallel model
    else:
        model_params = model.state_dict()

    # for param_tensor in model_params:
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    torch.save(
        {
            'model_state_dict': model_params,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'total_iter_num': total_iter_num,
            'epoch_loss': epoch_loss,
        }, filename)


def load_origin_Checkpoint(pathWeightsFile, model_name, model):
    '''Loads the origin weights of [model] from [pathWeightsFile]. 

        Args:
            pathWeightsFile (str): The path of checkpoint
            model_name (str): The model name ('densedepth', 'adabin' or 'dpt')
            model (torch.nn.Module): The model whose weights should be loaded

        Raises:
            ValueError: If the given file in path does not exist
            ValueError: If the given model is not supported

        Returns:
            torch.nn.Module): The model with loaded weights.
    '''

    if not os.path.isfile(pathWeightsFile):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(pathWeightsFile))

    if model_name == 'densedepth':
        # Load model into GPU / CPU
        custom_objects = {
            'BilinearUpSampling2D': tf.keras.layers.UpSampling2D, 'depth_loss_function': None}
        model_tf = tf.keras.models.load_model(
            pathWeightsFile, custom_objects=custom_objects, compile=False)
        names = [weight.name for layer in model_tf.layers for weight in layer.weights]
        weights = model_tf.get_weights()

        keras_name = []
        for name, weight in zip(names, weights):
          keras_name.append(name)

        model = model.float()

        # load parameter from keras
        keras_state_dict = {} 
        j = 0
        for name, param in model.named_parameters():
        
          if 'classifier' in name:
            keras_state_dict[name]=param
            continue
        
          if 'conv' in name and 'weight' in name:
            keras_state_dict[name]=torch.from_numpy(np.transpose(weights[j],(3, 2, 0, 1)))
            # print(name,keras_name[j])
            j = j+1
            continue
        
          if 'conv' in name and 'bias' in name:
            keras_state_dict[name]=torch.from_numpy(weights[j])
            # print(param.shape,weights[j].size)
            j = j+1
            continue
        
          if 'norm' in name and 'weight' in name:
            keras_state_dict[name]=torch.from_numpy(weights[j])
            # print(param.shape,weights[j].shape)
            j = j+1
            continue
        
          if 'norm' in name and 'bias' in name:
            keras_state_dict[name]=torch.from_numpy(weights[j])
            # print(param.shape,weights[j].size)
            j = j+1
            keras_state_dict[name.replace("bias", "running_mean")]=torch.from_numpy(weights[j])
            # print(param.shape,weights[j].size)
            j = j+1
            keras_state_dict[name.replace("bias", "running_var")]=torch.from_numpy(weights[j])
            # print(param.shape,weights[j].size)
            j = j+1
            continue


        model.load_state_dict(keras_state_dict)

        #custom_objects = {
        #    'BilinearUpSampling2D': tf.keras.layers.UpSampling2D, 'depth_loss_function': None}
        #model_tf = tf.keras.models.load_model(
        #    pathWeightsFile, custom_objects=custom_objects, compile=False)
        #keras_weights = dict()
        #for layer in model_tf.layers:
        #    if type(layer) is tf.keras.layers.Conv2D:
        #        keras_weights[layer.get_config(
        #        )['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
#
        #model_dict = model.state_dict()
        #pretrained_dict = {k: torch.from_numpy(
        #    v) for k, v in keras_weights.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #model.load_state_dict(model_dict)
    elif model_name == 'adabin':
        CHECKPOINT = torch.load(
            pathWeightsFile, map_location='cpu')
        load_dict = {}
        model_state = CHECKPOINT['model']
        for k, v in model_state.items():
            if k.startswith('module.'):
                k_ = k.replace('module.', '')
                load_dict[k_] = v
            else:
                load_dict[k] = v

        modified = {}  # backward compatibility to older naming of architecture blocks
        for k, v in load_dict.items():
            if k.startswith('adaptive_bins_layer.embedding_conv.'):
                k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                               'adaptive_bins_layer.conv3x3.')
                modified[k_] = v
                # del load_dict[k]

            elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

                k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                               'adaptive_bins_layer.patch_transformer.embedding_convPxP')
                modified[k_] = v
                # del load_dict[k]
            else:
                modified[k] = v  # else keep the original

        model.load_state_dict(modified)
    elif model_name == 'dpt':
        parameters = torch.load(pathWeightsFile, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        model.load_state_dict(parameters)
    elif model_name == 'lapdepth':
        model_state = torch.load(
            pathWeightsFile, map_location='cpu')

        load_dict = {}
        for k, v in model_state.items():
            if k.startswith('module.'):
                k_ = k[7:]
                load_dict[k_] = v
            else:
                load_dict[k] = v
       
        model.load_state_dict(load_dict)
    else:
        raise ValueError('Invalid origin is given for transfer learning.\
                The model {} is not supported'.format(model_name))

    print(colored('{}'.format(pathWeightsFile), 'green'))
    return model


def load_checkpoint(pathWeightsFile, model):
    '''Loads the trained weights of [model] from [pathWeightsFile] and returns the checkpoint and the model. 

        Args:
            pathWeightsFile (str): The path of checkpoint
            model (torch.nn.Module): The model whose weights should be loaded

        Raises:
            ValueError: If the given file in path does not exist

        Returns:
            dict(str,str): The model checkpoint
            torch.nn.Module): The model with loaded weights.
    '''

    print(colored(
        'Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
    if not os.path.isfile(pathWeightsFile):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(pathWeightsFile))

    CHECKPOINT = torch.load(pathWeightsFile, map_location='cpu')
    if 'model_state_dict' in CHECKPOINT:
        model_state = CHECKPOINT['model_state_dict']
        load_dict = {}
        for k, v in model_state.items():
            if k.startswith('module.'):
                k_ = k[7:]
                load_dict[k_] = v
            else:
                load_dict[k] = v
        model.load_state_dict(load_dict)

    print(colored('{}'.format(pathWeightsFile), 'green'))
    return CHECKPOINT, model

