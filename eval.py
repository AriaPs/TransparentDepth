'''

Evalidate the models for depth map estimation of transparent structures

Note: This file is adapted from ClearGrasp evaluation framework.

'''
import argparse
import csv
import errno
import os
import glob
import shutil

from termcolor import colored
import yaml
from attrdict import AttrDict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from tqdm import tqdm

import dataloader
import dataloader_clearGrasp
import dataloader_nyu
import dataloader_transDepth
from utils import framework_eval, model_io

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device_ids = [0]


def create_csv(csv_filename, csv_dir):
    '''Creates a csv file in the given path and returns a list of its header field names. 

        Args:
            csv_filename (str): The csv file name
            csv_dir (str): The csv save path

        Returns:
            List of str: A list of its header field names. 
    '''

    field_names = ["Model", "Image Num", "a1", "a2", "a3", "abs_rel",
                       "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
    with open(os.path.join(csv_dir, csv_filename), 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=field_names, delimiter=',')
        writer.writeheader()
    return field_names

def get_test_DataSet(config):
    '''Returns a dictionary containig all DataLoader classes for test data sets with respect to specified arguments 
        in the config yaml file. 

        Args:
            config (dict): The parsed configuration YAML file

        Returns:
            dict(str, DataLoader): A dictionary having the name of DataLoader as key and the DataLoader as value
    '''

    if(config.eval.dataset =="nyu"):
        return {'NYU Depth V2 Test Set': dataloader_nyu.NYUDepthDataLoader(config, 'test').data }
    elif(config.eval.dataset =="transDepth"):
        return {'TransDepth Test Set': dataloader_transDepth.TransDepthDataLoader(config.eval.transDepthDatasetTest, config.eval.batchSize, config.eval.numWorkers, 'test').data }
    else:
        return get_ClearGrasp_DataLoader(config)

     
def get_ClearGrasp_DataLoader(config):
    
    # Make new dataloaders for each synthetic dataset
    db_test_list_synthetic = []
    if config.eval.datasetsTestSynthetic is not None:
        for dataset in config.eval.datasetsTestSynthetic:
            print('Creating Synthetic Images dataset from: "{}"'.format(dataset.images))
            if dataset.images:
                db = dataloader_clearGrasp.ClearGraspsDataset(input_dir=dataset.images,
                                                    mask_dir= dataset.mask,
                                                   depth_dir=dataset.depths,
                                                   width=config.train.imgWidth,
                                                   height=config.train.imgHeight,
                                                   mode='test')
                db_test_list_synthetic.append(db)

    # Make new dataloaders for each real dataset
    db_test_list_real = []
    if config.eval.datasetsTestReal is not None:
        for dataset in config.eval.datasetsTestReal:
            print('Creating Real Images dataset from: "{}"'.format(dataset.images))
            if dataset.images:
                db = dataloader_clearGrasp.ClearGraspsDataset(input_dir=dataset.images,
                                                   mask_dir= dataset.mask,
                                                   depth_dir=dataset.depths,
                                                   width=config.train.imgWidth,
                                                   height=config.train.imgHeight,
                                                   mode='test',
                                                   isSynthetic=False)
                db_test_list_real.append(db)

    # Create pytorch dataloaders from datasets
    dataloaders_dict = {}
    if db_test_list_synthetic:
        db_test_synthetic = torch.utils.data.ConcatDataset(
            db_test_list_synthetic)
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

    return dataloaders_dict

def get_ClearGrasp_DataLoader_with_imgaug(config):
    augs_test = iaa.Sequential([
        iaa.Resize({
            "height": config.eval.imgHeight,
            "width": config.eval.imgWidth
        }, interpolation='nearest'),
    ])

    # Make new dataloaders for each synthetic dataset
    db_test_list_synthetic = []
    if config.eval.datasetsTestSynthetic is not None:
        for dataset in config.eval.datasetsTestSynthetic:
            print('Creating Synthetic Images dataset from: "{}"'.format(dataset.images))
            if dataset.images:
                db = dataloader.ClearGraspsDataset(input_dir=dataset.images,
                                                   depth_dir=dataset.depths,
                                                   transform=augs_test,
                                                   input_only=None)
                db_test_list_synthetic.append(db)

    # Make new dataloaders for each real dataset
    db_test_list_real = []
    if config.eval.datasetsTestReal is not None:
        for dataset in config.eval.datasetsTestReal:
            print('Creating Real Images dataset from: "{}"'.format(dataset.images))
            if dataset.images:
                db = dataloader.ClearGraspsDataset(input_dir=dataset.images,
                                                   depth_dir=dataset.depths,
                                                   transform=augs_test,
                                                   input_only=None,
                                                   isSynthetic=False)
                db_test_list_real.append(db)

    # Create pytorch dataloaders from datasets
    dataloaders_dict = {}
    if db_test_list_synthetic:
        db_test_synthetic = torch.utils.data.ConcatDataset(
            db_test_list_synthetic)
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

    return dataloaders_dict


if __name__ == '__main__':
    print('Depth Map Estimation of transparent structures. Loading checkpoint...')

    parser = argparse.ArgumentParser(
        description='Run eval of depth completion on synthetic and real test dataset')
    parser.add_argument('-c', '--configFile', required=True,
                        help='Path to config yaml file', metavar='path/to/config')
    args = parser.parse_args()

    ###################### Load Config File #############################
    CONFIG_FILE_PATH = args.configFile  # 'config/config.yaml'
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = AttrDict(config_yaml)

    # Create directory to save results
    SUBDIR_RESULT = 'results'
    SUBDIR_IMG = 'img_files'

    results_root_dir = config.eval.resultsDir
    runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    results_dir = os.path.join(
        results_root_dir, 'exp-{:03d}'.format(prev_run_id))
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

    csv_dir = os.path.join(results_dir, SUBDIR_RESULT)

    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('Saving results to folder: ' +
          colored('"{}"\n'.format(results_dir), 'blue'))

    ###################### DataLoader #############################

    dataloaders_dict = get_test_DataSet(config)

    ###################### ModelBuilder #############################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

    if (config.eval.compareResult and config.eval.dataset in ['clearGrasp','nyu']) or config.eval.densedepth.should_validate:
        from models.DenseDepth import DenseDepth
        # DenseDepth
        densedepth_model = DenseDepth()
        if config.eval.loadProjektCheckpoints:
            _ , densedepth_model = model_io.load_checkpoint(
            config.eval.densedepth.pathWeightsFile, densedepth_model)
        else:
            densedepth_model = model_io.load_origin_Checkpoint(config.eval.densedepth.pathWeightsFile, 'densedepth', densedepth_model) 
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            densedepth_model = nn.DataParallel(densedepth_model)
        densedepth_model = densedepth_model.to(device)

    if config.eval.compareResult or config.eval.adabin.should_validate:
        from models.AdaBin import UnetAdaptiveBins
        # Adabin
        adabin_model = UnetAdaptiveBins.build(n_bins=config.eval.adabin.n_bins, min_val=config.eval.min_depth,
                                              max_val=config.eval.max_depth, norm=config.eval.adabin.norm)
        if config.eval.loadProjektCheckpoints:
            _ , adabin_model = model_io.load_checkpoint(
            config.eval.adabin.pathWeightsFile, adabin_model)
        else:
            adabin_model = model_io.load_origin_Checkpoint(config.eval.adabin.pathWeightsFile, 'adabin', adabin_model)
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            adabin_model = nn.DataParallel(adabin_model)
        adabin_model = adabin_model.to(device)

    if config.eval.compareResult or config.eval.dpt.should_validate:
        from models.DPT.models import DPTDepthModel
        dpt_model = DPTDepthModel(
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        if config.eval.loadProjektCheckpoints:
            _ , dpt_model = model_io.load_checkpoint(
            config.eval.dpt.pathWeightsFile, dpt_model)
        else:
            dpt_model = model_io.load_origin_Checkpoint(config.eval.dpt.pathWeightsFile, 'dpt', dpt_model)
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            dpt_model = nn.DataParallel(dpt_model)
        dpt_model = dpt_model.to(device)
    
    if (config.eval.compareResult and config.eval.dataset  in ['clearGrasp','nyu']) or config.eval.lapdepth.should_validate:
        from models.LapDepth import LDRN
        lapdepth_model = LDRN({
                'lv6': False,
                'encoder': 'ResNext101',
                'norm': 'BN',
                'act': 'ReLU',
                'max_depth': config.eval.max_depth,
            })
        
        if config.eval.loadProjektCheckpoints:
            _ , lapdepth_model = model_io.load_checkpoint(
            config.eval.lapdepth.pathWeightsFile, lapdepth_model)
        else:
            lapdepth_model = model_io.load_origin_Checkpoint(config.eval.lapdepth.pathWeightsFile, 'lapdepth', lapdepth_model)
        
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            lapdepth_model = nn.DataParallel(lapdepth_model)
        lapdepth_model = lapdepth_model.to(device)

    if (config.eval.compareResult and config.eval.dataset =='transDepth') or config.eval.newcrf.should_validate:
        from models.NewCRFDepth.NewCRFDepth import NewCRFDepth
        newCRF_model = NewCRFDepth(version='large07', inv_depth=False, max_depth=config.eval.max_depth, pretrained=None)
        
        if config.eval.loadProjektCheckpoints:
            _ , newCRF_model = model_io.load_checkpoint(
            config.eval.newcrf.pathWeightsFile, newCRF_model)
        else:
            newCRF_model = model_io.load_origin_Checkpoint(config.eval.newcrf.pathWeightsFile, 'newcrf', newCRF_model)
        
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            newCRF_model = nn.DataParallel(newCRF_model)
        newCRF_model = newCRF_model.to(device)

    if (config.eval.compareResult and config.eval.dataset =='transDepth') or config.eval.glp.should_validate:
        from models.GLPDepth.model import GLPDepth
        glp_model = GLPDepth(max_depth=config.eval.max_depth, is_train=False)
        
        if config.eval.loadProjektCheckpoints:
            _ , glp_model = model_io.load_checkpoint(config.eval.glp.pathWeightsFile, glp_model)
        else:
            glp_model = model_io.load_origin_Checkpoint(config.eval.glp.pathWeightsFile, 'glp', glp_model)
        
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            glp_model = nn.DataParallel(glp_model)
        glp_model = glp_model.to(device)
        

    if (config.eval.compareResult and config.eval.dataset =='transDepth') or config.eval.depthformer.should_validate:
        from depth.models import build_depther
        import mmcv
        cfg = mmcv.Config.fromfile(config.eval.depthformer.modelPath)
        depthformer_model = build_depther(
        cfg.model,
        train_cfg=None,
        test_cfg=None)
        #model.init_weights()
        del cfg
        
        if config.eval.loadProjektCheckpoints:
            _ , depthformer_model = model_io.load_checkpoint(
            config.eval.depthformer.pathWeightsFile, depthformer_model)
        else:
            depthformer_model = model_io.load_origin_Checkpoint(config.eval.depthformer.pathWeightsFile, 'depthformer', depthformer_model)
        
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            depthformer_model = nn.DataParallel(depthformer_model)
        depthformer_model = depthformer_model.to(device)

    
    if (config.eval.compareResult and config.eval.dataset =='transDepth') or config.eval.binsformer.should_validate:
        from depth.models import build_depther
        import mmcv
        cfg = mmcv.Config.fromfile(config.eval.binsformer.modelPath)
        binsformer_model = build_depther(
        cfg.model,
        train_cfg=None,
        test_cfg=None)
        del cfg
        
        if config.eval.loadProjektCheckpoints:
            _ , binsformer_model = model_io.load_checkpoint(
            config.eval.binsformer.pathWeightsFile, binsformer_model)
        else:
            binsformer_model = model_io.load_origin_Checkpoint(config.eval.binsformer.pathWeightsFile, 'depthformer', binsformer_model)
        
        # Enable Multi-GPU training
        if torch.cuda.device_count() > 1:
            binsformer_model = nn.DataParallel(binsformer_model)
        binsformer_model = binsformer_model.to(device)

    
    ###################### Eval #############################

    if config.eval.compareResult:
        # if in compare modus compare all models

        # Create CSV File to store error metrics
        csv_filename = 'computed_errors_exp_{:03d}.csv'.format(prev_run_id)

        field_names = create_csv(csv_filename, csv_dir)

        if config.eval.dataset  in ['clearGrasp','nyu']:
            framework_eval.validateAll_old(adabin_model, densedepth_model, dpt_model, lapdepth_model, device, config, dataloaders_dict,
                                   field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG)
        else:
            framework_eval.validateAll(adabin_model, dpt_model, glp_model, depthformer_model, newCRF_model, binsformer_model, device, config, dataloaders_dict,
                                   field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG)
    else:
        # else compare only the specified models
        if config.eval.densedepth.should_validate:

            # Create CSV File to store error metrics
            csv_filename = 'densedepth_computed_errors_exp_{:03d}.csv'.format(prev_run_id)

            field_names = create_csv(csv_filename, csv_dir)

            framework_eval.validateDenseDepht(densedepth_model, device, config, dataloaders_dict,
                                              field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG)

        elif config.eval.adabin.should_validate:

            # Create CSV File to store error metrics
            csv_filename = 'adabin_computed_errors_exp_{:03d}.csv'.format(prev_run_id)

            field_names = create_csv(csv_filename, csv_dir)

            framework_eval.validateAdaBin(adabin_model, device, config, dataloaders_dict,
                                          field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG)
        
        elif config.eval.dpt.should_validate:

            # Create CSV File to store error metrics
            csv_filename = 'dpt_computed_errors_exp_{:03d}.csv'.format(prev_run_id)

            field_names = create_csv(csv_filename, csv_dir)

            framework_eval.validateDPT(dpt_model, device, config, dataloaders_dict,
                                          field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG) 
        
        elif config.eval.lapdepth.should_validate:

            # Create CSV File to store error metrics
            csv_filename = 'lapdepth_computed_errors_exp_{:03d}.csv'.format(prev_run_id)

            field_names = create_csv(csv_filename, csv_dir)

            framework_eval.validateFullResolutionModel("LapDepth",lapdepth_model, device, config, dataloaders_dict,
                                          field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG) 
                                        
        elif config.eval.newcrf.should_validate:

            # Create CSV File to store error metrics
            csv_filename = 'dpt_computed_errors_exp_{:03d}.csv'.format(prev_run_id)

            field_names = create_csv(csv_filename, csv_dir)

            framework_eval.validateFullResolutionModel("NewCRF", newCRF_model, device, config, dataloaders_dict,
                                          field_names, csv_filename, csv_dir, results_dir, SUBDIR_IMG) 
