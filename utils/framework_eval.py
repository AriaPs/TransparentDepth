

'''

Evaluation framework for depth map estimation of transparent structures.

Note: This file is adapted from ClearGrasp evaluation framework.

'''

import os
import csv

from numpy.core.fromnumeric import resize

import imageio
import numpy as np
import imgaug as ia
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from termcolor import colored
from imgaug import augmenters as iaa

from utils.api import depth2rgb, exr_saver
import loss_functions


def validateAll(model_dpt, depthformer_model, newCRF_model, device, config, dataloaders_dict, field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG):
    '''Computes the standard evaluation metrics for [model_adabin], [model_densedepth], [model_dpt] from inputs taken of data set 
    listed in [dataloaders_dict] and saves it in the csv file [csv_filename].
    The results are saved in [results_dir] and the images in [results_dir]/[SUBDIR_IMG].

        Args:
            model_adabin (torch.nn.Module): The model instance of AdaBins
            model_densedepth (torch.nn.Module): The model instance of DenseDepth
            model_dpt (torch.nn.Module): The model instance of DPT
            device (list of int): List of GPU ids on which the model is trained
            config (dict): The parsed configuration YAML file
            dataloaders_dict (dict(str, DataLoader)): A dictionary having the name of DataLoader as key and the DataLoader as value
            field_names (list of str): A list of csv header field names. 
            csv_filename (str): Csv file name
            results_dir (str): The output path
            SUBDIR_IMG (str): The path where the img outputs are saved

    '''

    model_dpt.eval()
    newCRF_model.eval()
    depthformer_model.eval()
    criterion =loss_functions.SILogLoss()
    metas = getMeta()
    #criterion_ssim = loss_functions.ssim

    ### Run Validation and Test Set ###
    print('\nCompare Validation of all Models')
    print('-' * 50 + '\n')
    print(colored('Results will be saved to: {}\n'.format(
        config.eval.resultsDir), 'green'))

    for key in dataloaders_dict:
        print('Running on {} dataset:'.format(key))
        print('=' * 30)

        running_loss_dpt = 0.0
        metric_dpt = createMetricDict()
        metric_dpt_masked = createMetricDict()
        metric_dpt_masked_opaque = createMetricDict()
        metric_dpt_masked_forground = createMetricDict()
        metric_dpt_masked_background = createMetricDict()

        running_loss_depthformer = 0.0
        metric_depthformer = createMetricDict()
        metric_depthformer_masked = createMetricDict()
        metric_depthformer_masked_opaque = createMetricDict()
        metric_depthformer_masked_forground = createMetricDict()
        metric_depthformer_masked_background = createMetricDict()

        running_loss_newcrf = 0.0
        metric_newcrf = createMetricDict()
        metric_newcrf_masked = createMetricDict()
        metric_newcrf_masked_opaque = createMetricDict()
        metric_newcrf_masked_forground = createMetricDict()
        metric_newcrf_masked_background = createMetricDict()

        testLoader = dataloaders_dict[key]
        for ii, sample_batched in enumerate(tqdm(testLoader)):

            inputs, mask, mask_forground, depths = sample_batched

            depths = depths.to(device)

            # Forward pass of the mini-batch
            with torch.no_grad():
                input = inputs.to(device)
                model_output_newcrf = newCRF_model(input)
                model_output_dpt = model_dpt(input)
                depthformer_model_result = depthformer_model(return_loss=True, depth_gt=depths, img=input, img_metas=metas)
            
                #loss_depthformer = depthformer_model_result['decode.loss_depth']
                model_output_depthformer = depthformer_model_result['decode.depth_pred']

                # [BXHXW] --> [BX1XHXW]
                model_output_dpt = torch.unsqueeze(model_output_dpt, 1)

            # Compute the loss

            loss_newcrf = criterion(model_output_newcrf, depths, mask=mask, interpolate= False)
            running_loss_newcrf += loss_newcrf.item()

            loss_depthformer = criterion(model_output_depthformer, depths, mask=mask)
            running_loss_depthformer += loss_depthformer.item()

            loss_dpt = criterion(model_output_dpt, depths, mask=mask, interpolate= False)
            running_loss_dpt += loss_dpt.item()

            model_output_depthformer = resize_pred(model_output_depthformer, depths.shape[-2:], config)

            # Save output images, one at a time, to results
            img_tensor = inputs.detach()
            output_tensor_dpt = model_output_dpt.detach().cpu()
            output_tensor_depthformer = model_output_depthformer.detach().cpu()
            output_tensor_newcrf = model_output_newcrf.detach().cpu()
            depth_tensor = depths.detach().cpu()
            mask_tensor = mask.detach().cpu()
            mask_forground_tensor = mask_forground.detach().cpu()

            # Extract each tensor within batch and save results
            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor_dpt, output_tensor_depthformer, output_tensor_newcrf, depth_tensor, mask_tensor, mask_forground_tensor)):
                img, output_dpt, output_depthformer, output_newcrf, gt, mask, mask_forground = sample_batched
                
                # Calc metrics
                metric_dpt, batch_metric_dpt = update_metric(metric_dpt, output_dpt, gt, config.eval.dataset)
                metric_depthformer, batch_metric_depthformer = update_metric(metric_depthformer, output_depthformer, gt, config.eval.dataset)
                metric_newcrf, batch_metric_newcrf = update_metric(metric_newcrf, output_newcrf, gt, config.eval.dataset)

                metric_dpt_masked, batch_metric_dpt_masked = update_metric(metric_dpt_masked, output_dpt, gt, config.eval.dataset, mask=mask)
                metric_depthformer_masked, batch_metric_depthformer_masked = update_metric(metric_depthformer_masked, output_depthformer, gt, config.eval.dataset, mask=mask)
                metric_newcrf_masked, batch_metric_newcrf_masked = update_metric(metric_newcrf_masked, output_newcrf, gt, config.eval.dataset, mask=mask)

                metric_dpt_masked_opaque, batch_metric_dpt_masked_opaque = update_metric(metric_dpt_masked_opaque, output_dpt, gt, config.eval.dataset, mask=mask, masksZeros=False)
                metric_depthformer_masked_opaque, batch_metric_depthformer_masked_opaque = update_metric(metric_depthformer_masked_opaque, output_depthformer, gt, config.eval.dataset, mask=mask, masksZeros=False)
                metric_newcrf_masked_opaque, batch_metric_newcrf_masked_opaque = update_metric(metric_newcrf_masked_opaque, output_newcrf, gt, config.eval.dataset, mask=mask, masksZeros=False)

                metric_dpt_masked_forground, batch_metric_dpt_masked_forground = update_metric(metric_dpt_masked_forground, output_dpt, gt, config.eval.dataset, mask=mask_forground)
                metric_depthformer_masked_forground, batch_metric_depthformer_masked_forground = update_metric(metric_depthformer_masked_forground, output_depthformer, gt, config.eval.dataset, mask=mask_forground)
                metric_newcrf_masked_forground, batch_metric_newcrf_masked_forground = update_metric(metric_newcrf_masked_forground, output_newcrf, gt, config.eval.dataset, mask=mask_forground)

                metric_dpt_masked_background, batch_metric_dpt_masked_background = update_metric(metric_dpt_masked_background, output_dpt, gt, config.eval.dataset, mask=mask_forground, masksZeros=False)
                metric_depthformer_masked_background, batch_metric_depthformer_masked_background = update_metric(metric_depthformer_masked_background, output_depthformer, gt, config.eval.dataset, mask=mask_forground, masksZeros=False)
                metric_newcrf_masked_background, batch_metric_newcrf_masked_background = update_metric(metric_newcrf_masked_background, output_newcrf, gt, config.eval.dataset, mask=mask_forground, masksZeros=False)

                # Write the data into a csv file
                write_csv_row("DPT", config.eval.batchSize, batch_metric_dpt,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DepthFormer", config.eval.batchSize, batch_metric_depthformer,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("NewCRF", config.eval.batchSize, batch_metric_newcrf,
                              csv_dir, field_names, csv_filename, ii, iii)

                write_csv_row("DPT masked: Trans", config.eval.batchSize, batch_metric_dpt_masked,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DepthFormer masked: Trans", config.eval.batchSize, batch_metric_depthformer_masked,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("NewCRF masked: Trans", config.eval.batchSize, batch_metric_newcrf_masked,
                              csv_dir, field_names, csv_filename, ii, iii)

                write_csv_row("DPT masked: Opaque", config.eval.batchSize, batch_metric_dpt_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DepthFormer masked: Opaque", config.eval.batchSize, batch_metric_depthformer_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("NewCRF masked: Opaque", config.eval.batchSize, batch_metric_newcrf_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)

                write_csv_row("DPT masked: forground", config.eval.batchSize, batch_metric_dpt_masked_forground,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DepthFormer masked: forground", config.eval.batchSize, batch_metric_depthformer_masked_forground,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("NewCRF masked: forground", config.eval.batchSize, batch_metric_newcrf_masked_forground,
                              csv_dir, field_names, csv_filename, ii, iii)

                write_csv_row("DPT masked: background", config.eval.batchSize, batch_metric_dpt_masked_background,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DepthFormer masked: background", config.eval.batchSize, batch_metric_depthformer_masked_background,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("NewCRF masked: background", config.eval.batchSize, batch_metric_newcrf_masked_background,
                              csv_dir, field_names, csv_filename, ii, iii)

                if config.eval.saveCompareImage:
                    save_compare_images(img, output_dpt, output_depthformer, output_newcrf,
                                    gt, config, results_dir, SUBDIR_IMG, ii, iii, key)

        num_batches = len(testLoader)  # Num of batches
        num_images = len(testLoader.dataset)  # Num of total images
        print('\nnum_batches:', num_batches)
        print('num_images:', num_images)
        epoch_loss_dpt = running_loss_dpt / num_batches
        epoch_loss_depthformer = running_loss_depthformer / num_batches
        epoch_loss_newcrf = running_loss_newcrf / num_batches
        print('Test Mean Loss DPT: {:.4f} \n'.format(epoch_loss_dpt))
        print('Test Mean Loss DepthFormer: {:.4f} \n'.format(epoch_loss_depthformer))
        print('Test Mean Loss newcrf: {:.4f} \n'.format(epoch_loss_newcrf))

        print_means(metric_dpt, num_images, "DPT", csv_dir, csv_filename, field_names)
        print_means(metric_depthformer, num_images, "DepthFormer", csv_dir, csv_filename, field_names)
        print_means(metric_newcrf, num_images, "NewCRF", csv_dir, csv_filename, field_names)

        print_means(metric_dpt_masked, num_images, "DPT masked: Trans", csv_dir, csv_filename, field_names)
        print_means(metric_depthformer_masked, num_images, "DepthFormer masked: Trans", csv_dir, csv_filename, field_names)
        print_means(metric_newcrf_masked, num_images, "NewCRF masked: Trans", csv_dir, csv_filename, field_names)

        print_means(metric_dpt_masked_opaque, num_images, "DPT masked: Opaque", csv_dir, csv_filename, field_names)
        print_means(metric_depthformer_masked_opaque, num_images, "DepthFormer masked: Opaque", csv_dir, csv_filename, field_names)
        print_means(metric_newcrf_masked_opaque, num_images, "NewCRF masked: Opaque", csv_dir, csv_filename, field_names)

        print_means(metric_dpt_masked_forground, num_images, "DPT masked: forground", csv_dir, csv_filename, field_names)
        print_means(metric_depthformer_masked_forground, num_images, "DepthFormer masked: forground", csv_dir, csv_filename, field_names)
        print_means(metric_newcrf_masked_forground, num_images, "NewCRF masked: forground", csv_dir, csv_filename, field_names)

        print_means(metric_dpt_masked_background, num_images, "DPT masked: background", csv_dir, csv_filename, field_names)
        print_means(metric_depthformer_masked_background, num_images, "DepthFormer masked: background", csv_dir, csv_filename, field_names)
        print_means(metric_newcrf_masked_background, num_images, "NewCRF masked: background", csv_dir, csv_filename, field_names)

def validateAll_old(model_adabin, model_densedepth, model_dpt, model_lapdepth, device, config, dataloaders_dict, field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG):
    '''Computes the standard evaluation metrics for [model_adabin], [model_densedepth], [model_dpt] from inputs taken of data set 
    listed in [dataloaders_dict] and saves it in the csv file [csv_filename].
    The results are saved in [results_dir] and the images in [results_dir]/[SUBDIR_IMG].

        Args:
            model_adabin (torch.nn.Module): The model instance of AdaBins
            model_densedepth (torch.nn.Module): The model instance of DenseDepth
            model_dpt (torch.nn.Module): The model instance of DPT
            device (list of int): List of GPU ids on which the model is trained
            config (dict): The parsed configuration YAML file
            dataloaders_dict (dict(str, DataLoader)): A dictionary having the name of DataLoader as key and the DataLoader as value
            field_names (list of str): A list of csv header field names. 
            csv_filename (str): Csv file name
            results_dir (str): The output path
            SUBDIR_IMG (str): The path where the img outputs are saved

    '''

    model_adabin.eval()
    model_densedepth.eval()
    model_dpt.eval()
    model_lapdepth.eval()
    criterion =loss_functions.SILogLoss()
    #criterion_ssim = loss_functions.ssim

    ### Run Validation and Test Set ###
    print('\nCompare Validation of all Models')
    print('-' * 50 + '\n')
    print(colored('Results will be saved to: {}\n'.format(
        config.eval.resultsDir), 'green'))

    for key in dataloaders_dict:
        print('Running on {} dataset:'.format(key))
        print('=' * 30)

        running_loss_adabin = 0.0
        metric_adabin = createMetricDict()
        metric_adabin_masked = createMetricDict()
        metric_adabin_masked_opaque = createMetricDict()

        running_loss_densedepth = 0.0
        metric_densedepth = createMetricDict()
        metric_densedepth_masked = createMetricDict()
        metric_densedepth_masked_opaque = createMetricDict()

        running_loss_dpt = 0.0
        metric_dpt = createMetricDict()
        metric_dpt_masked = createMetricDict()
        metric_dpt_masked_opaque = createMetricDict()

        running_loss_lapdepth = 0.0
        metric_lapdepth = createMetricDict()
        metric_lapdepth_masked = createMetricDict()
        metric_lapdepth_masked_opaque = createMetricDict()

        testLoader = dataloaders_dict[key]
        for ii, sample_batched in enumerate(tqdm(testLoader)):

            inputs, mask, depths = sample_batched

            depths = depths.to(device)

            # Forward pass of the mini-batch
            with torch.no_grad():
                input = inputs.to(device)
                model_output_densedepth = model_densedepth(input)
                _ , model_output_adabin = model_adabin(input)
                _ , model_output_lapdepth = model_lapdepth(input)
                model_output_dpt = model_dpt(input)
                # [BXHXW] --> [BX1XHXW]
                model_output_dpt = torch.unsqueeze(model_output_dpt, 1)

            # Compute the loss

            loss_adabin = criterion(model_output_adabin, depths, mask=mask)

            running_loss_adabin += loss_adabin.item()

            loss_densedepth = criterion(model_output_densedepth, depths, mask=mask)
            running_loss_densedepth += loss_densedepth.item()

            loss_lapdepth = criterion(model_output_lapdepth, depths, mask=mask)
            running_loss_lapdepth += loss_lapdepth.item()

            loss_dpt = criterion(model_output_dpt, depths, mask=mask, interpolate= False)
            running_loss_dpt += loss_dpt.item()

            
            model_output_adabin = resize_pred(model_output_adabin, depths.shape[-2:], config)
            model_output_densedepth = resize_pred(model_output_densedepth, depths.shape[-2:], config)

            # Save output images, one at a time, to results
            img_tensor = inputs.detach()
            output_tensor_adabin = model_output_adabin.detach().cpu()
            output_tensor_densedepth = model_output_densedepth.detach().cpu()
            output_tensor_dpt = model_output_dpt.detach().cpu()
            output_tensor_lapdepth = model_output_lapdepth.detach().cpu()
            depth_tensor = depths.detach().cpu()
            mask_tensor = mask.detach().cpu()
            #depths_resized_tensor = depths_resized.detach().cpu()

            # Extract each tensor within batch and save results
            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor_adabin, output_tensor_densedepth, output_tensor_dpt, output_tensor_lapdepth, depth_tensor, mask_tensor)):
                img, output_adabin, output_densedepth, output_dpt, output_lapdepth, gt, mask = sample_batched
                
                if(not config.eval.loadProjektCheckpoints):
                    output_densedepth = np.clip(config.eval.max_depth/output_densedepth, config.eval.min_depth, config.eval.max_depth)

                # Calc metrics
                metric_adabin, batch_metric_adabin = update_metric(metric_adabin, output_adabin, gt, config.eval.dataset)
                metric_densedepth, batch_metric_densedepth = update_metric(metric_densedepth, output_densedepth, gt, config.eval.dataset)
                metric_dpt, batch_metric_dpt = update_metric(metric_dpt, output_dpt, gt, config.eval.dataset)
                metric_lapdepth, batch_metric_lapdepth = update_metric(metric_lapdepth, output_lapdepth, gt, config.eval.dataset)

                metric_adabin_masked, batch_metric_adabin_masked = update_metric(metric_adabin_masked, output_adabin, gt, config.eval.dataset, mask=mask)
                metric_densedepth_masked, batch_metric_densedepth_masked = update_metric(metric_densedepth_masked, output_densedepth, gt, config.eval.dataset, mask=mask)
                metric_dpt_masked, batch_metric_dpt_masked = update_metric(metric_dpt_masked, output_dpt, gt, config.eval.dataset, mask=mask)
                metric_lapdepth_masked, batch_metric_lapdepth_masked = update_metric(metric_lapdepth_masked, output_lapdepth, gt, config.eval.dataset, mask=mask)

                metric_adabin_masked_opaque, batch_metric_adabin_masked_opaque = update_metric(metric_adabin_masked_opaque, output_adabin, gt, config.eval.dataset, mask=mask, maskOpaques=False)
                metric_densedepth_masked_opaque, batch_metric_densedepth_masked_opaque = update_metric(metric_densedepth_masked_opaque, output_densedepth, gt, config.eval.dataset, mask=mask, maskOpaques=False)
                metric_dpt_masked_opaque, batch_metric_dpt_masked_opaque = update_metric(metric_dpt_masked_opaque, output_dpt, gt, config.eval.dataset, mask=mask, maskOpaques=False)
                metric_lapdepth_masked_opaque, batch_metric_lapdepth_masked_opaque = update_metric(metric_lapdepth_masked_opaque, output_lapdepth, gt, config.eval.dataset, mask=mask, maskOpaques=False)

                # Write the data into a csv file
                write_csv_row("Adabin", config.eval.batchSize, batch_metric_adabin,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("Adabin masked: Trans", config.eval.batchSize, batch_metric_adabin_masked,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("Adabin masked: Opaque", config.eval.batchSize, batch_metric_adabin_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DenseDepth", config.eval.batchSize, batch_metric_densedepth,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DenseDepth masked: Trans", config.eval.batchSize, batch_metric_densedepth_masked,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DenseDepth masked: Opaue", config.eval.batchSize, batch_metric_densedepth_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DPT", config.eval.batchSize, batch_metric_dpt,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DPT masked: Trans", config.eval.batchSize, batch_metric_dpt_masked,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("DPT masked: Opaque", config.eval.batchSize, batch_metric_dpt_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("LapDepth", config.eval.batchSize, batch_metric_lapdepth,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("LapDepth masked: Trans", config.eval.batchSize, batch_metric_lapdepth_masked,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("LapDepth masked: Opaque", config.eval.batchSize, batch_metric_lapdepth_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)

                if config.eval.saveCompareImage:
                    save_compare_images_old(img, output_adabin, output_densedepth, output_dpt, output_lapdepth,
                                    gt, config, results_dir, SUBDIR_IMG, ii, iii, key)

        num_batches = len(testLoader)  # Num of batches
        num_images = len(testLoader.dataset)  # Num of total images
        print('\nnum_batches:', num_batches)
        print('num_images:', num_images)
        epoch_loss_adabin = running_loss_adabin / num_batches
        epoch_loss_densedepth = running_loss_densedepth / num_batches
        epoch_loss_dpt = running_loss_dpt / num_batches
        epoch_loss_lapdepth = running_loss_lapdepth / num_batches
        print('Test Mean Loss Adabin: {:.4f} \n'.format(epoch_loss_adabin))
        print('Test Mean Loss DenseDepth: {:.4f} \n'.format(
            epoch_loss_densedepth))
        print('Test Mean Loss DPT: {:.4f} \n'.format(epoch_loss_dpt))
        print('Test Mean Loss LapDepth: {:.4f} \n'.format(epoch_loss_lapdepth))

        print_means(metric_adabin, num_images, "Adabin", csv_dir, csv_filename, field_names)
        print_means(metric_densedepth, num_images, "DenseDepth", csv_dir, csv_filename, field_names)
        print_means(metric_dpt, num_images, "DPT", csv_dir, csv_filename, field_names)
        print_means(metric_lapdepth, num_images, "LapDepth", csv_dir, csv_filename, field_names)

        print_means(metric_adabin_masked, num_images, "Adabin masked: Trans", csv_dir, csv_filename, field_names)
        print_means(metric_densedepth_masked, num_images, "DenseDepth masked: Trans", csv_dir, csv_filename, field_names)
        print_means(metric_dpt_masked, num_images, "DPT masked: Trans", csv_dir, csv_filename, field_names)
        print_means(metric_lapdepth_masked, num_images, "LapDepth masked: Trans", csv_dir, csv_filename, field_names)

        print_means(metric_adabin_masked_opaque, num_images, "Adabin masked: Opaque", csv_dir, csv_filename, field_names)
        print_means(metric_densedepth_masked_opaque, num_images, "DenseDepth masked: Opaque", csv_dir, csv_filename, field_names)
        print_means(metric_dpt_masked_opaque, num_images, "DPT masked: Opaque", csv_dir, csv_filename, field_names)
        print_means(metric_lapdepth_masked_opaque, num_images, "LapDepth masked: Opaque", csv_dir, csv_filename, field_names)

def validateAdaBin(model, device, config, dataloaders_dict, field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG):
    '''Computes the standard evaluation metrics for [model] from inputs taken of data set 
    listed in [dataloaders_dict] and saves it in the csv file [csv_filename].
    The results are saved in [results_dir] and the images in [results_dir]/[SUBDIR_IMG].

        Args:
            model (torch.nn.Module): The model instance 
            device (list of int): List of GPU ids on which the model is trained
            config (dict): The parsed configuration YAML file
            dataloaders_dict (dict(str, DataLoader)): A dictionary having the name of DataLoader as key and the DataLoader as value
            field_names (list of str): A list of csv header field names. 
            csv_filename (str): Csv file name
            results_dir (str): The output path
            SUBDIR_IMG (str): The path where the img outputs are saved

    '''

    model.eval()

    criterion = loss_functions.ssim

    ### Run Validation and Test Set ###
    print('\nValidation - AdaBin Model')
    print('-' * 50 + '\n')
    print(colored('Results will be saved to: {}\n'.format(
        config.eval.resultsDir), 'green'))

    for key in dataloaders_dict:
        print('Running AdaBin on {} dataset:'.format(key))
        print('=' * 30)

        running_loss = 0.0
        
        metric_adabin = createMetricDict()

        testLoader = dataloaders_dict[key]
        for ii, sample_batched in enumerate(tqdm(testLoader)):

            inputs, depths = sample_batched
            
            # Forward pass of the mini-batch
            with torch.no_grad():
               _ , model_output = model(inputs.to(device))

            loss = criterion(model_output, depths.to(device))
            running_loss += loss.item()

            model_output = resize_pred(model_output, depths.shape[-2:], config)

            # Save output images, one at a time, to results
            img_tensor = inputs.detach()
            output_tensor = model_output.detach().cpu()
            depth_tensor = depths.detach()

            # Extract each tensor within batch and save results
            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, depth_tensor)):
                img, output, gt = sample_batched
                
                # Calc metrics
                metric_adabin, batch_metric_adabin = update_metric(metric_adabin, output, gt, config.eval.dataset)

                # Write the data into a csv file
                write_csv_row("AdaBin", config.eval.batchSize, batch_metric_adabin,
                              csv_dir, field_names, csv_filename, ii, iii)

                if config.eval.adabin.saveImgae:
                    save_images(img, output, gt, config,
                                results_dir, SUBDIR_IMG, ii, iii, key)

        num_batches = len(testLoader)  # Num of batches
        num_images = len(testLoader.dataset)  # Num of total images
        print('\nnum_batches:', num_batches)
        print('num_images:', num_images)
        epoch_loss = running_loss / num_batches
        print('Test Mean Loss: {:.4f}'.format(epoch_loss))

        print_means(metric_adabin, num_images, "AdaBin", csv_dir, csv_filename, field_names)

def validateDenseDepht(model, device, config, dataloaders_dict, field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG):
    '''Computes the standard evaluation metrics for [model] from inputs taken of data set 
    listed in [dataloaders_dict] and saves it in the csv file [csv_filename].
    The results are saved in [results_dir] and the images in [results_dir]/[SUBDIR_IMG].

        Args:
            model (torch.nn.Module): The model instance 
            device (list of int): List of GPU ids on which the model is trained
            config (dict): The parsed configuration YAML file
            dataloaders_dict (dict(str, DataLoader)): A dictionary having the name of DataLoader as key and the DataLoader as value
            field_names (list of str): A list of csv header field names. 
            csv_filename (str): Csv file name
            results_dir (str): The output path
            SUBDIR_IMG (str): The path where the img outputs are saved

    '''

    model.eval()

    criterion = loss_functions.ssim

    ### Run Validation and Test Set ###
    print('\nValidation - DenseDepth Model')
    print('-' * 50 + '\n')
    print(colored('Results will be saved to: {}\n'.format(
        config.eval.resultsDir), 'green'))

    for key in dataloaders_dict:
        print('Running DenseDepth on {} dataset:'.format(key))
        print('=' * 30)

        running_loss = 0.0
        
        metric_densedepth = createMetricDict()

        testLoader = dataloaders_dict[key]
        for ii, sample_batched in enumerate(tqdm(testLoader)):

            inputs, depths = sample_batched

            # Forward pass of the mini-batch
            with torch.no_grad():
                model_output = model(inputs.to(device))

            loss = criterion(model_output, depths.to(device))
            running_loss += loss.item()

            model_output = resize_pred(model_output, depths.shape[-2:], config)

            # Save output images, one at a time, to results
            img_tensor = inputs.detach()
            output_tensor = model_output.detach().cpu()
            depth_tensor = depths.detach()

            # Extract each tensor within batch and save results
            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, depth_tensor)):
                img, output, gt = sample_batched

                if(not config.eval.loadProjektCheckpoints):
                    output = np.clip(config.eval.max_depth/output, config.eval.min_depth, config.eval.max_depth) / config.eval.max_depth

                # Calc metrics
                metric_densedepth, batch_metric_densedepth = update_metric(metric_densedepth, output, gt, config.eval.dataset) 

                # Write the data into a csv file
                write_csv_row("DenseDepth", config.eval.batchSize, batch_metric_densedepth,
                              csv_dir, field_names, csv_filename, ii, iii)

                if config.eval.densedepth.saveImgae:
                    save_images(img, output, gt, config,
                                results_dir, SUBDIR_IMG, ii, iii, key)

        num_batches = len(testLoader)  # Num of batches
        num_images = len(testLoader.dataset)  # Num of total images
        print('\nnum_batches:', num_batches)
        print('num_images:', num_images)
        epoch_loss = running_loss / num_batches
        print('Test Mean Loss: {:.4f}'.format(epoch_loss))

        print_means(metric_densedepth, num_images, "DenseDepth", csv_dir, csv_filename, field_names)

def validateFullResolutionModel(modelName, model, device, config, dataloaders_dict, field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG):
    '''Computes the standard evaluation metrics for [model] from inputs taken of data set 
    listed in [dataloaders_dict] and saves it in the csv file [csv_filename].
    The results are saved in [results_dir] and the images in [results_dir]/[SUBDIR_IMG].

        Args:
            model (torch.nn.Module): The model instance 
            device (list of int): List of GPU ids on which the model is trained
            config (dict): The parsed configuration YAML file
            dataloaders_dict (dict(str, DataLoader)): A dictionary having the name of DataLoader as key and the DataLoader as value
            field_names (list of str): A list of csv header field names. 
            csv_filename (str): Csv file name
            results_dir (str): The output path
            SUBDIR_IMG (str): The path where the img outputs are saved

    '''

    model.eval()

    criterion = loss_functions.ssim

    ### Run Validation and Test Set ###
    print('\nValidation - ', modelName, ' Model')
    print('-' * 50 + '\n')
    print(colored('Results will be saved to: {}\n'.format(
        config.eval.resultsDir), 'green'))

    for key in dataloaders_dict:
        print('Running ',modelName,' on {} dataset:'.format(key))
        print('=' * 30)

        running_loss = 0.0
        
        metric = createMetricDict()
        metric_masked = createMetricDict()
        metric_masked_opaque = createMetricDict()

        testLoader = dataloaders_dict[key]
        for ii, sample_batched in enumerate(tqdm(testLoader)):

            inputs, mask, depths = sample_batched
            
            # Forward pass of the mini-batch
            with torch.no_grad():
               _ , model_output = model(inputs.to(device))

            loss = criterion(model_output, depths.to(device))
            running_loss += loss.item()

            
            # Save output images, one at a time, to results
            img_tensor = inputs.detach()
            output_tensor = model_output.detach().cpu()
            depth_tensor = depths.detach()

            # Extract each tensor within batch and save results
            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, depth_tensor)):
                img, output, gt = sample_batched
                
                # Calc metrics
                metric, batch_metric = update_metric(metric, output, gt, config.eval.dataset)
                metric_masked, batch_metric_masked = update_metric(metric_masked, output, gt, config.eval.dataset, mask=mask)
                metric_masked_opaque, batch_metric_masked_opaque = update_metric(metric_masked_opaque, output, gt, config.eval.dataset, mask=mask, maskOpaques=False)

                # Write the data into a csv file
                write_csv_row(modelName, config.eval.batchSize, batch_metric,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("masked: Trans", config.eval.batchSize, batch_metric_masked,
                              csv_dir, field_names, csv_filename, ii, iii)
                write_csv_row("masked: Opaque", config.eval.batchSize, batch_metric_masked_opaque,
                              csv_dir, field_names, csv_filename, ii, iii)

                if config.eval.lapdepth.saveImgae:
                    save_images(img, output, gt, config,
                                results_dir, SUBDIR_IMG, ii, iii, key)

        num_batches = len(testLoader)  # Num of batches
        num_images = len(testLoader.dataset)  # Num of total images
        print('\nnum_batches:', num_batches)
        print('num_images:', num_images)
        epoch_loss = running_loss / num_batches
        print('Test Mean Loss: {:.4f}'.format(epoch_loss))

        print_means(metric, num_images, modelName, csv_dir, csv_filename, field_names)
        print_means(metric_masked, num_images, "masked: Trans", csv_dir, csv_filename, field_names)
        print_means(metric_masked_opaque, num_images, "masked: Opaque", csv_dir, csv_filename, field_names)

def validateDPT(model, device, config, dataloaders_dict, field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG):
    '''Computes the standard evaluation metrics for [model] from inputs taken of data set 
    listed in [dataloaders_dict] and saves it in the csv file [csv_filename].
    The results are saved in [results_dir] and the images in [results_dir]/[SUBDIR_IMG].

        Args:
            model (torch.nn.Module): The model instance 
            device (list of int): List of GPU ids on which the model is trained
            config (dict): The parsed configuration YAML file
            dataloaders_dict (dict(str, DataLoader)): A dictionary having the name of DataLoader as key and the DataLoader as value
            field_names (list of str): A list of csv header field names. 
            csv_filename (str): Csv file name
            results_dir (str): The output path
            SUBDIR_IMG (str): The path where the img outputs are saved

    '''

    model.eval()

    criterion = loss_functions.ssim

    ### Run Validation and Test Set ###
    print('\nValidation - DPT Model')
    print('-' * 50 + '\n')
    print(colored('Results will be saved to: {}\n'.format(
        config.eval.resultsDir), 'green'))

    for key in dataloaders_dict:
        print('Running DPT on {} dataset:'.format(key))
        print('=' * 30)

        running_loss = 0.0
        
        metric_dpt = createMetricDict()

        testLoader = dataloaders_dict[key]
        for ii, sample_batched in enumerate(tqdm(testLoader)):

            inputs, depths = sample_batched

            # Forward pass of the mini-batch
            with torch.no_grad():
               model_output = model(inputs.to(device))
               # [BXHXW] --> [BX1XHXW]
               model_output = torch.unsqueeze(model_output, 1)

            loss = criterion(model_output, depths.to(device), interpolate=False)
            running_loss += loss.item()

            # Save output images, one at a time, to results
            img_tensor = inputs.detach()
            output_tensor = model_output.detach().cpu()
            depth_tensor = depths.detach()

            # Extract each tensor within batch and save results
            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, depth_tensor)):
                img, output, gt = sample_batched
                
                # Calc metrics
                metric_dpt, batch_metric_dpt = update_metric(metric_dpt, output, gt, config.eval.dataset)

                # Write the data into a csv file
                write_csv_row("DPT", config.eval.batchSize, batch_metric_dpt,
                              csv_dir, field_names, csv_filename, ii, iii)

                if config.eval.dpt.saveImgae:
                    save_images(img, output, gt, config,
                                results_dir, SUBDIR_IMG, ii, iii, key)

        num_batches = len(testLoader)  # Num of batches
        num_images = len(testLoader.dataset)  # Num of total images
        print('\nnum_batches:', num_batches)
        print('num_images:', num_images)
        epoch_loss = running_loss / num_batches
        print('Test Mean Loss: {:.4f}'.format(epoch_loss))

        print_means(metric_dpt, num_images, "DPT", csv_dir, csv_filename, field_names)

def createMetricDict():
    '''Returns a dictionary saving the evaluation metrics.

        Returns:
            Dict(str, list of float): Dictionary saving the evaluation metrics
    '''

    metric_a1 = 0
    metric_a2 = 0
    metric_a3 = 0
    metric_abs_rel = 0
    metric_rmse = 0
    metric_log_10 = 0
    metric_rmse_log = 0
    metric_si_log = 0
    metric_sq_rel = 0
    return dict(metric_a1=metric_a1, metric_a2=metric_a2, metric_a3=metric_a3, metric_abs_rel=metric_abs_rel, metric_rmse=metric_rmse, metric_log_10=metric_log_10, metric_rmse_log=metric_rmse_log,
                metric_si_log=metric_si_log, metric_sq_rel=metric_sq_rel)

def print_means(metric, num_images, model_name, results_dir, csv_filename, field_names):
    '''Computes the epoch means of evaluation metrics and saves it in the csv file [csv_filename].
    Moreover, it prints the result in console.

        Args:
            metric (dict): The metric dict having stored the evaluation metrics
            num_images (int): Number of images for wich the metrics are computed
            model_name (str): The name of evaluated model
            results_dir (str): The path where the outputs are saved
            csv_filename (str): The csv file name
            field_names (list of str): The csv header field names
    '''
 
    epoch_metric_a1 = metric['metric_a1'] / num_images
    epoch_metric_a2 = metric['metric_a2'] / num_images
    epoch_metric_a3 = metric['metric_a3'] / num_images
    epoch_metric_abs_rel = metric['metric_abs_rel'] / num_images
    epoch_metric_rmse = metric['metric_rmse'] / num_images
    epoch_metric_log_10 = metric['metric_log_10'] / num_images
    epoch_metric_rmse_log = metric['metric_rmse_log'] / num_images
    epoch_metric_si_log = metric['metric_si_log'] / num_images
    epoch_metric_sq_rel = metric['metric_sq_rel'] / num_images  

    
    # Write the data into a csv file
    with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=field_names, delimiter=',')
        #"Image Num", "a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log","silog", "sq_rel"
        row_data = {'Model': model_name,
                    'Image Num': 'MEAN',
                    'a1': epoch_metric_a1,
                    'a2': epoch_metric_a2,
                    'a3': epoch_metric_a3,
                    'abs_rel': epoch_metric_abs_rel,
                    'rmse': epoch_metric_rmse,
                    'log_10': epoch_metric_log_10,
                    'rmse_log': epoch_metric_rmse_log,
                    'silog': epoch_metric_si_log,
                    'sq_rel': epoch_metric_sq_rel}
        writer.writerow(row_data)

    print(
        '\nTest Mean Metrics {} - \n a1: {:.2f}, a2: {:.2f}, a3: {:.2f}, abs_rel: {:.2f}, rmse: {:.2f} \n log_10: {:.2f},rmse_log: {:.2f},si_log: {:.2f},sq_rel: {:.2f} \n num_images: {}\n\n'
        .format(model_name, epoch_metric_a1, epoch_metric_a2, epoch_metric_a3, epoch_metric_abs_rel, epoch_metric_rmse,
                epoch_metric_log_10, epoch_metric_rmse_log, epoch_metric_si_log, epoch_metric_sq_rel, num_images))

def update_metric(metricDict, output, gt, dataset= "clearGrasp", mask=None, masksZeros=True):
    '''Computes the evaluation metrics for [output] and [gt] and updates [metricDict]. Finally, 
    it returns the updated [metricDict] and the computed metrics. 

        Args:
            metricDict (Dict(str, list of float)): Dictionary saving the evaluation metrics
            output (Torch.Tensor): The network output
            gt (Torch.Tensor): The ground truth

        Returns:
            Dict(str, list of float): Updated dictionary having the evaluation metrics
            list of float: The computed metrics
    '''
    metrics = loss_functions.compute_errors(gt, output, dataset=dataset, masks=mask, masksZeros=masksZeros)
    metricDict['metric_a1'] += metrics['a1']
    metricDict['metric_a2'] += metrics['a2']
    metricDict['metric_a3'] += metrics['a3']
    metricDict['metric_abs_rel'] += metrics['abs_rel']
    metricDict['metric_rmse'] += metrics['rmse']
    metricDict['metric_log_10'] += metrics['log_10']
    metricDict['metric_rmse_log'] += metrics['rmse_log']
    metricDict['metric_si_log'] += metrics['silog']
    metricDict['metric_sq_rel'] += metrics['sq_rel']
    return metricDict, metrics

def save_images(input_image, output, gt, config, results_dir, SUBDIR_IMG, dataLoader_index, batch_index, set_name):
    '''Generates for [input_image], [output] and [gt] a grid image having their depth map visualizations and 
    saves it in  [results_dir]/[SUBDIR_IMG].

        Args:
            input_image (Torch.Tensor): Dictionary saving the evaluation metrics
            output (Torch.Tensor): The network output
            gt (Torch.Tensor): The ground truth
            config (dict): The parsed configuration YAML file
            results_dir (str): The path where the outputs are saved
            SUBDIR_IMG (str): The name of folder where images are saved
            dataLoader_index (int): The index of a batch in Dataloader
            batch_index (int): The index of image in a batch
            set_name (str): The name of test set (Real or Synthtetic)

    '''

    size = (int(config.eval.imgWidth/2), int(config.eval.imgHeight/2))
        
    # Save PNG 
    output_rgb = depth2rgb(output[0])
    output_rgb = cv2.resize(
        output_rgb, size, interpolation=cv2.INTER_LINEAR)

    gt_rgb = depth2rgb(gt[0])
    gt_rgb = cv2.resize(gt_rgb, size, interpolation=cv2.INTER_LINEAR)

    gt_output_path_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                      '{:09d}-{}-img-gt-output.png'.format(dataLoader_index * config.eval.batchSize + batch_index, set_name))

    

    img = cv2.normalize(input_image.numpy().transpose(1, 2, 0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_rgb = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    grid_image = np.concatenate((img_rgb, gt_rgb, output_rgb), 1)
    imageio.imwrite(gt_output_path_rgb, grid_image)

def save_compare_images(input_image, output_dpt, output_depthformer, output_newcrf, gt, config, results_dir, SUBDIR_IMG, dataLoader_index, batch_index, set_name, norm=5):
    '''Generates for  [input_image], [gt], [output_densedepth], [output_adabin], and [output_dpt] a grid image having their depth map visualizations and 
    saves it in  [results_dir]/[SUBDIR_IMG].

        Args:
            input_image (Torch.Tensor): Dictionary saving the evaluation metrics
            output_adabin (Torch.Tensor): The network output of AdaBins
            output_densedepth (Torch.Tensor): The network output of DenseDepth
            output_dpt (Torch.Tensor): The network output of DPT
            gt (Torch.Tensor): The ground truth
            config (dict): The parsed configuration YAML file
            results_dir (str): The path where the outputs are saved
            SUBDIR_IMG (str): The name of folder where images are saved
            dataLoader_index (int): The index of a batch in Dataloader
            batch_index (int): The index of image in a batch
            set_name (str): The name of test set (Real or Synthtetic)

    '''
    size = (config.eval.imgWidth, config.eval.imgHeight)

    # Save PNG

    output_rgb_dpt = depth2rgb(output_dpt[0])
    output_rgb_dpt = cv2.resize(
        output_rgb_dpt, size, interpolation=cv2.INTER_LINEAR)

    output_rgb_output_depthformer = depth2rgb(output_depthformer[0])
    output_rgb_output_depthformer = cv2.resize(
        output_rgb_output_depthformer, size, interpolation=cv2.INTER_LINEAR)

    output_rgb_newcrf = depth2rgb(output_newcrf[0])
    output_rgb_newcrf = cv2.resize(
        output_rgb_newcrf, size, interpolation=cv2.INTER_LINEAR)

    

    gt_rgb = depth2rgb(gt[0])
    gt_rgb = cv2.resize(gt_rgb, size, interpolation=cv2.INTER_LINEAR)
    
    img = cv2.normalize(input_image.numpy().transpose(1, 2, 0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    name_prefix = dataLoader_index * config.eval.batchSize + batch_index

    ##### save png files

    img_gt_output_paths_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                       '{:09d}-{}-img-gt-outputs.png'.format(name_prefix, set_name))

    grid_image = np.concatenate(
        (img, gt_rgb, output_rgb_dpt, output_rgb_output_depthformer, output_rgb_newcrf), 1)

    imageio.imwrite(img_gt_output_paths_rgb, grid_image)

    if config.eval.saveNormedImg:

        output_dpt_normed = torch.nn.functional.normalize(output_dpt[0]) * norm
        output_rgb_dpt_normed = depth2rgb(output_dpt_normed)
        output_rgb_dpt_normed = cv2.resize(
            output_rgb_dpt_normed, size, interpolation=cv2.INTER_LINEAR)

        output_depthformer_normed = torch.nn.functional.normalize(output_depthformer[0]) * norm
        output_rgb_depthformer_normed = depth2rgb(output_depthformer_normed)
        output_rgb_depthformer_normed = cv2.resize(
            output_rgb_depthformer_normed, size, interpolation=cv2.INTER_LINEAR)

        output_newcrf_normed = torch.nn.functional.normalize(output_newcrf[0]) * norm
        output_rgb_newcrf_normed = depth2rgb(output_newcrf_normed)
        output_rgb_newcrf_normed = cv2.resize(
            output_rgb_newcrf_normed, size, interpolation=cv2.INTER_LINEAR)

        img_gt_output_paths_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-img-gt-outputs-normed.png'.format(name_prefix, set_name))

        gt_rgb = depth2rgb(torch.nn.functional.normalize(gt[0])) * norm
        gt_rgb = cv2.resize(gt_rgb, size, interpolation=cv2.INTER_LINEAR)

        grid_image = np.concatenate(
            (img, gt_rgb, output_rgb_dpt_normed, output_rgb_depthformer_normed, output_rgb_newcrf_normed), 1)

        imageio.imwrite(img_gt_output_paths_rgb, grid_image)

    ##### save exr files
    if config.eval.saveEXR:
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-gt-normed.exr'.format(name_prefix, set_name))
                                           
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-newcrf-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, output_newcrf_normed.numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-newcrf.exr'.format(name_prefix, set_name))
                                           
        exr_saver(save_path_exr, output_newcrf[0].numpy())

        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-depthformer-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, output_depthformer_normed.numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-depthformer.exr'.format(name_prefix, set_name))
                                           
        exr_saver(save_path_exr, output_depthformer[0].numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-dpt-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, output_dpt_normed.numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-dpt.exr'.format(name_prefix, set_name))
                                           
        exr_saver(save_path_exr, output_dpt[0].numpy())
 
def save_compare_images_old(input_image, output_adabin, output_densedepth, output_dpt, output_lapdepth, gt, config, results_dir, SUBDIR_IMG, dataLoader_index, batch_index, set_name, norm=4):
    '''Generates for  [input_image], [gt], [output_densedepth], [output_adabin], and [output_dpt] a grid image having their depth map visualizations and 
    saves it in  [results_dir]/[SUBDIR_IMG].

        Args:
            input_image (Torch.Tensor): Dictionary saving the evaluation metrics
            output_adabin (Torch.Tensor): The network output of AdaBins
            output_densedepth (Torch.Tensor): The network output of DenseDepth
            output_dpt (Torch.Tensor): The network output of DPT
            gt (Torch.Tensor): The ground truth
            config (dict): The parsed configuration YAML file
            results_dir (str): The path where the outputs are saved
            SUBDIR_IMG (str): The name of folder where images are saved
            dataLoader_index (int): The index of a batch in Dataloader
            batch_index (int): The index of image in a batch
            set_name (str): The name of test set (Real or Synthtetic)

    '''
    size = (config.eval.imgWidth, config.eval.imgHeight)

    # Save PNG
    output_rgb_output_adabin = depth2rgb(output_adabin[0])
    output_rgb_output_adabin = cv2.resize(
        output_rgb_output_adabin, size, interpolation=cv2.INTER_LINEAR)

    output_rgb_densedepth = depth2rgb(output_densedepth[0])
    output_rgb_densedepth = cv2.resize(
        output_rgb_densedepth, size, interpolation=cv2.INTER_LINEAR)

    output_rgb_dpt = depth2rgb(output_dpt[0])
    output_rgb_dpt = cv2.resize(
        output_rgb_dpt, size, interpolation=cv2.INTER_LINEAR)

    output_rgb_lapdepth = depth2rgb(output_lapdepth[0])
    output_rgb_lapdepth = cv2.resize(
        output_rgb_lapdepth, size, interpolation=cv2.INTER_LINEAR)

    

    gt_rgb = depth2rgb(gt[0])
    gt_rgb = cv2.resize(gt_rgb, size, interpolation=cv2.INTER_LINEAR)
    
    img = cv2.normalize(input_image.numpy().transpose(1, 2, 0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    name_prefix = dataLoader_index * config.eval.batchSize + batch_index

    ##### save png files

    img_gt_output_paths_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                       '{:09d}-{}-img-gt-outputs.png'.format(name_prefix, set_name))

    grid_image = np.concatenate(
        (img, gt_rgb, output_rgb_densedepth, output_rgb_lapdepth, output_rgb_output_adabin, output_rgb_dpt), 1)

    imageio.imwrite(img_gt_output_paths_rgb, grid_image)

    if config.eval.saveNormedImg:

        output_adabin_normed = torch.nn.functional.normalize(output_adabin[0]) * norm
        output_rgb_output_adabin_normed = depth2rgb(output_adabin_normed)
        output_rgb_output_adabin_normed = cv2.resize(
            output_rgb_output_adabin_normed, size, interpolation=cv2.INTER_LINEAR)
    
        output_densedepth_normed = torch.nn.functional.normalize(output_densedepth[0]) * norm
        output_rgb_densedepth_normed = depth2rgb(output_densedepth_normed)
        output_rgb_densedepth_normed = cv2.resize(
            output_rgb_densedepth_normed, size, interpolation=cv2.INTER_LINEAR)


        output_dpt_normed = torch.nn.functional.normalize(output_dpt[0]) * norm
        output_rgb_dpt_normed = depth2rgb(output_dpt_normed)
        output_rgb_dpt_normed = cv2.resize(
            output_rgb_dpt_normed, size, interpolation=cv2.INTER_LINEAR)


        output_lapdepth_normed = torch.nn.functional.normalize(output_lapdepth[0]) * norm
        output_rgb_lapdepth_normed = depth2rgb(output_lapdepth_normed)
        output_rgb_lapdepth_normed = cv2.resize(
            output_rgb_lapdepth_normed, size, interpolation=cv2.INTER_LINEAR)

        img_gt_output_paths_rgb = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-img-gt-outputs-normed.png'.format(name_prefix, set_name))

        gt_rgb = depth2rgb(torch.nn.functional.normalize(gt[0])) * norm
        gt_rgb = cv2.resize(gt_rgb, size, interpolation=cv2.INTER_LINEAR)

        grid_image = np.concatenate(
            (img, gt_rgb, output_rgb_densedepth_normed, output_rgb_lapdepth_normed, output_rgb_output_adabin_normed, output_rgb_dpt_normed), 1)

        imageio.imwrite(img_gt_output_paths_rgb, grid_image)

    ##### save exr files
    if config.eval.saveEXR:
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-gt-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, gt[0].numpy()/1.5)
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-adabin-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, output_adabin_normed.numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-adabin.exr'.format(name_prefix, set_name))
                                           
        exr_saver(save_path_exr, output_adabin[0].numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-densedepth-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, output_densedepth_normed.numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-densedepth.exr'.format(name_prefix, set_name))
                                           
        exr_saver(save_path_exr, output_densedepth[0].numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-lapdepth-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, output_lapdepth_normed.numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-lapdepth.exr'.format(name_prefix, set_name))
                                           
        exr_saver(save_path_exr, output_lapdepth[0].numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-dpt-normed.exr'.format(name_prefix, set_name))
    
        exr_saver(save_path_exr, output_dpt_normed.numpy())
    
        save_path_exr = os.path.join(results_dir, SUBDIR_IMG,
                                           '{:09d}-{}-depth-dpt.exr'.format(name_prefix, set_name))
                                           
        exr_saver(save_path_exr, output_dpt[0].numpy())
    
def write_csv_row(model_name, batchSize, metrics, results_dir, field_names, csv_filename, dataLoader_index, batch_index):
    '''Writes a row in the csv file [csv_filename] having the computed [metrics].

        Args:
            model_name (str): The name of the evaluated model
            batchSize (int): The batch size
            metrics (list of float): The computed evaluation metrics
            results_dir (str): The path where the outputs are saved
            csv_filename (str): The csv file name
            field_names (list of str): The csv header field names
            dataLoader_index (int): The index of a batch in Dataloader
            batch_index (int): The index of image in a batch

    '''
    # Write the data into a csv file
    with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=field_names, delimiter=',')
        #"Image Num", "a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log","silog", "sq_rel"
        row_data = {'Model': model_name,
                    'Image Num': ((dataLoader_index * batchSize) + batch_index),
                    'a1': metrics['a1'],
                    'a2': metrics['a2'],
                    'a3': metrics['a3'],
                    'abs_rel': metrics['abs_rel'],
                    'rmse': metrics['rmse'],
                    'log_10': metrics['log_10'],
                    'rmse_log': metrics['rmse_log'],
                    'silog': metrics['silog'],
                    'sq_rel': metrics['sq_rel']}
        writer.writerow(row_data)

# Resize Tensor

def resize_pred(pred, gt_shape, config):
    '''Resizes the [pred] to the shape of ground truth ([gt_shape]).

        Args:
            pred (Torch.Tensor): A tensor which should be resized
            gt_shape ((BXCXHXW)): Shape of ground truth 
            config (dict): The parsed configuration YAML file

        Returns:
            Torch.Tensor: The resized tensor
    '''
    pred = nn.functional.interpolate(pred, gt_shape, mode='bilinear', align_corners=True)
    pred = torch.clamp(pred, config.eval.min_depth, config.eval.max_depth)
    return pred

def getMeta():
    results = dict(img=None)
    results['filename'] = None
    results['ori_filename'] = None
    results['pad_shape'] = None
    results['scale_factor'] = None
    results['flip'] = None
    results['flip_direction'] = None
    results['img_norm_cfg'] = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
    results['cam_intrinsic'] = None
    results['img_shape'] = (512,512,3)
    results['ori_shape'] = (512,512,3)
    return results

def resize_tensor(input_tensor, height, width):
    '''Resize the [input_tensor] to the given [height] and [width].

        Args:
            input_tensor (Torch.Tensor): The input tensor which going to be resized
            height (int): The new height
            width (int): The new width
        
        Returns:
            Torch.Tensor: The resized tensor

    '''
    augs_depth_resize = iaa.Sequential(
        [iaa.Resize({"height": height, "width": width}, interpolation='nearest')])
    det_tf = augs_depth_resize.to_deterministic()
    input_tensor = input_tensor.numpy().transpose(0, 2, 3, 1)
    resized_array = det_tf.augment_images(input_tensor)
    resized_array = torch.from_numpy(resized_array.transpose(0, 3, 1, 2))
    resized_array = resized_array.type(torch.DoubleTensor)

    return resized_array

