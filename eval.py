'''

Evalidate the models for depth map estimation of transparent structures

TODO: DOC

'''
import argparse
import csv
import errno
import os
import glob
import io
import shutil

from termcolor import colored
import yaml
from attrdict import AttrDict
import imageio
import numpy as np
import h5py
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

import model
import dataloader
import loss_functions
from utils import utils

print('Depth Map Estimation of transparent structures. Loading checkpoint...')

parser = argparse.ArgumentParser(
    description='Run eval of depth completion on synthetic data')
parser.add_argument('-c', '--configFile', required=True,
                    help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

###################### Load Config File #############################
CONFIG_FILE_PATH = args.configFile  # 'config/config.yaml'
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

###################### Load Checkpoint and its data #############################
if not os.path.isfile(config.eval.pathWeightsFile):
    raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
        config.eval.pathWeightsFile))

# Read config file stored in the model checkpoint to re-use it's params
CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location='cpu')
if 'model_state_dict' in CHECKPOINT:
    print(colored('Loaded data from checkpoint {}'.format(
        config.eval.pathWeightsFile), 'green'))

    config_checkpoint_dict = CHECKPOINT['config']
    config_checkpoint = AttrDict(config_checkpoint_dict)
else:
    raise ValueError('The checkpoint file does not have model_state_dict in it.\
                     Please use the newer checkpoint files!')

# Create directory to save results
SUBDIR_RESULT = 'results'
SUBDIR_IMG = 'img_files'

results_root_dir = config.eval.resultsDir
runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
if os.path.isdir(os.path.join(results_dir, SUBDIR_RESULT)):
    NUM_FILES_IN_EMPTY_FOLDER = 0
    if len(os.listdir(os.path.join(results_dir, SUBDIR_RESULT))) > NUM_FILES_IN_EMPTY_FOLDER:
        prev_run_id += 1
        results_dir = os.path.join(
            results_root_dir, 'exp-{:03d}'.format(prev_run_id))
        os.makedirs(results_dir)
else:
    os.makedirs(results_dir)

try:
    os.makedirs(os.path.join(results_dir, SUBDIR_RESULT))
    os.makedirs(os.path.join(results_dir, SUBDIR_IMG))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('Saving results to folder: ' +
      colored('"{}"\n'.format(results_dir), 'blue'))

# Create CSV File to store error metrics
csv_filename = 'computed_errors_exp_{:03d}.csv'.format(prev_run_id)
field_names = ["Image Num", "Mean", "Median", "<11.25", "<22.5", "<30"]
with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
    writer.writeheader()

###################### DataLoader #############################
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": config.eval.imgHeight,
        "width": config.eval.imgWidth
    }, interpolation='nearest'),
])

# Make new dataloaders for each synthetic dataset
db_test_list_synthetic = []
if config.eval.datasetsSynthetic is not None:
    for dataset in config.eval.datasetsSynthetic:
        print('Creating Synthetic Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.ClearGraspsDataset(input_dir=dataset.images,
                                               depth_dir=dataset.depths,
                                               transform=augs_test,
                                               input_only=None)
            db_test_list_synthetic.append(db)

# Make new dataloaders for each real dataset
db_test_list_real = []
if config.eval.datasetsReal is not None:
    for dataset in config.eval.datasetsReal:
        print('Creating Real Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.ClearGraspsDataset(input_dir=dataset.images,
                                               depth_dir=dataset.depths,
                                               transform=augs_test,
                                               input_only=None)
            db_test_list_real.append(db)


# Create pytorch dataloaders from datasets
dataloaders_dict = {}
if db_test_list_synthetic:
    db_test_synthetic = torch.utils.data.ConcatDataset(db_test_list_synthetic)
    testLoader_synthetic = DataLoader(db_test_synthetic,
                                      batch_size=config.eval.batchSize,
                                      shuffle=False,
                                      num_workers=config.eval.numWorkers,
                                      drop_last=False)
    dataloaders_dict.update({'synthetic': testLoader_synthetic})

if db_test_list_real:
    db_test_real = torch.utils.data.ConcatDataset(db_test_list_real)
    testLoader_real = DataLoader(db_test_real,
                                 batch_size=config.eval.batchSize,
                                 shuffle=False,
                                 num_workers=config.eval.numWorkers,
                                 drop_last=False)
    dataloaders_dict.update({'real': testLoader_real})


assert (len(dataloaders_dict) >
        0), 'No valid datasets given in config.yaml to run inference on!'


# Resize Tensor
def resize_tensor(input_tensor, height, width):
    augs_depth_resize = iaa.Sequential(
        [iaa.Resize({"height": height, "width": width}, interpolation='nearest')])
    det_tf = augs_depth_resize.to_deterministic()
    input_tensor = input_tensor.numpy().transpose(0, 2, 3, 1)
    resized_array = det_tf.augment_images(input_tensor)
    resized_array = torch.from_numpy(resized_array.transpose(0, 3, 1, 2))
    resized_array = resized_array.type(torch.DoubleTensor)

    return resized_array


###################### ModelBuilder #############################
if config.eval.model == 'densedepth':
    model = model.Model()

model_state = CHECKPOINT['model_state_dict']
new_model_state = {}
for key in model_state.keys():
    new_model_state[key[7:]] = model_state[key]

model.load_state_dict(new_model_state)

# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

### Select Loss Func ###
if config_checkpoint.train.lossFunc == 'cosine':
    criterion = loss_functions.loss_fn_cosine
elif config_checkpoint.train.lossFunc == 'radians':
    criterion = loss_functions.loss_fn_radians
else:
    raise ValueError('Invalid lossFunc from config file. Can only be "cosine" or "radians".\
                     Value passed is: {}'.format(config_checkpoint.train.lossFunc))

### Run Validation and Test Set ###
print('\nInference - Surface Normal Estimation')
print('-' * 50 + '\n')
print(colored('Results will be saved to: {}\n'.format(
    config.eval.resultsDir), 'green'))

for key in dataloaders_dict:
    print('Running inference on {} dataset:'.format(key))
    print('=' * 30)

    running_loss = 0.0
    running_mean = []
    running_median = []
    running_percentage1 = []
    running_percentage2 = []
    running_percentage3 = []

    testLoader = dataloaders_dict[key]
    for ii, sample_batched in enumerate(tqdm(testLoader)):
        # NOTE: In raw data, invalid surface normals are represented by [-1, -1, -1]. However, this causes
        #       problems during normalization of vectors. So they are represented as [0, 0, 0] in our dataloader output.

        inputs, depths = sample_batched

        if config.eval.model == 'densedepth':
            depths_resized = resize_tensor(depths, int(
                depths.shape[2] / 2), int(depths.shape[3] / 2))
            depths_resized = depths_resized.to(device).double()

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        depths = depths.to(device)

        with torch.no_grad():
            model_output = model(inputs)

        loss = criterion(model_output, depths_resized,
                         reduction='elementwise_mean')
        running_loss += loss.item()

        # loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
        #     model_output_norm, depths)
        # print('Batch {:09d} Mean: {:.4f} deg'.format(ii, loss_deg_mean.item()))
        # print('Batch {:09d} Median: {:.4f} deg'.format(ii, loss_deg_median.item()))
        # print('Batch {:09d} P1: {:.4f} %'.format(ii, percentage_1.item()))
        # print('Batch {:09d} P2: {:.4f} %'.format(ii, percentage_2.item()))
        # print('Batch {:09d} P3: {:.4f} %'.format(ii, percentage_3.item()))
        # print('Batch {:09d} num_images: {:d}'.format(ii, depths.shape[0]))
        
        # Save output images, one at a time, to results
        img_tensor = inputs.detach().cpu()
        output_tensor = model_output.detach().cpu()
        depth_tensor = depths.detach().cpu()
        depths_resized = depths_resized.detach().cpu()

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, depth_tensor, depths_resized)):
            img, output, depth, depths_resized = sample_batched
            
            # Calc metrics
            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels = loss_functions.metric_calculator(
                output, depths_resized)
            running_mean.append(loss_deg_mean.item())
            running_median.append(loss_deg_median.item())
            running_percentage1.append(percentage_1.item())
            running_percentage2.append(percentage_2.item())
            running_percentage3.append(percentage_3.item())

            # Write the data into a csv file
            with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=field_names, delimiter=',')
                row_data = [((ii * config.eval.batchSize) + iii),
                            loss_deg_mean.item(),
                            loss_deg_median.item(),
                            percentage_1.item(),
                            percentage_2.item(),
                            percentage_3.item()]
                writer.writerow(dict(zip(field_names, row_data)))

            
            # Save PNG and EXR Output
            output_rgb = utils.depth2rgb(output[0].numpy())
            output_rgb = cv2.resize(
                output_rgb, (512, 288), interpolation=cv2.INTER_LINEAR)
            output_path_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-depth.png'.format(ii * config.eval.batchSize + iii))
            output_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-depth.exr'.format(ii * config.eval.batchSize + iii))

            #imageio.imwrite(output_path_rgb, output_rgb)
            utils.exr_saver(output_path_exr, output[0].numpy())

            # Save PNG and EXR Output
            gt_rgb = utils.depth2rgb(depth[0].numpy())
            gt_rgb = cv2.resize(
                gt_rgb, (512, 288), interpolation=cv2.INTER_LINEAR)
            gt_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-depth-gt.exr'.format(ii * config.eval.batchSize + iii))

            #imageio.imwrite(gt_path_rgb, gt_rgb)
            utils.exr_saver(gt_path_exr, depth[0].numpy())

            
            img = img.numpy().transpose(1, 2, 0)
            img = cv2.resize(
                img, (512, 288), interpolation=cv2.INTER_LINEAR)
            img_path_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-img.png'.format(ii * config.eval.batchSize + iii))

            imageio.imwrite(img_path_rgb, img)

            gt_output_path_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-gt-output.png'.format(ii * config.eval.batchSize + iii))
            
            grid_image = np.concatenate((gt_rgb, output_rgb), 1)
            imageio.imwrite(gt_output_path_rgb, grid_image)



    num_batches = len(testLoader)  # Num of batches
    num_images = len(testLoader.dataset)  # Num of total images
    print('\nnum_batches:', num_batches)
    print('num_images:', num_images)
    epoch_loss = running_loss / num_batches
    print('Test Mean Loss: {:.4f}'.format(epoch_loss))

    epoch_mean = sum(running_mean) / num_images
    epoch_median = sum(running_median) / num_images
    epoch_percentage1 = sum(running_percentage1) / num_images
    epoch_percentage2 = sum(running_percentage2) / num_images
    epoch_percentage3 = sum(running_percentage3) / num_images
    print(
        '\nTest Metrics - Mean: {:.2f}deg, Median: {:.2f}deg, P1: {:.2f}%, P2: {:.2f}%, p3: {:.2f}%, num_images: {}\n\n'
        .format(epoch_mean, epoch_median, epoch_percentage1, epoch_percentage2, epoch_percentage3, num_images))
