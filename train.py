'''

Training script for depth map estimation of transparent structures.

Note: This file is adapted from ClearGrasp and AdaBins trainig framework.

'''

from utils import framework_train, model_io
import dataloader
import dataloader_clearGrasp
import dataloader_transDepth
import loss_functions
from tqdm import tqdm
from torch.utils.data import DataLoader
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from imgaug import augmenters as iaa
from attrdict import AttrDict
import torch.nn as nn
import torch
import oyaml
import argparse
import errno
import glob
import io
import os
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device_ids = [0]


def loadDataSet_with_imaug(config):
    '''Returns the DataLoader classes for training and validation data sets with respect to specified arguments 
        in the config yaml file. 

        Args:
            config (dict): The parsed configuration YAML file

        Returns:
            DataLoader: DataLoader for trainig data set
            DataLoader: DataLoader for synthetic validation data set
            DataLoader: DataLoader for real validation data set
    '''

    augs_train = iaa.Sequential([
        # Geometric Augs
        iaa.Resize({
            "height": config.train.imgHeight,
            "width": config.train.imgWidth
        }, interpolation='nearest'),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Rot90((0, 4)),

        # Bright Patches
        iaa.Sometimes(
            0.1,
            iaa.blend.BlendAlpha(factor=(0.2, 0.7),
                            foreground=iaa.blend.BlendAlphaSimplexNoise(foreground=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                              upscale_method='cubic',
                                                              iterations=(1, 2)),
                            name="simplex-blend")),

        # Color Space Mods
        iaa.Sometimes(
            0.3,
            iaa.OneOf([
                iaa.Add((20, 20), per_channel=0.7, name="add"),
                iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
                iaa.WithColorspace(to_colorspace="HSV",
                                   from_colorspace="RGB",
                                   children=iaa.WithChannels(
                                       0, iaa.Add((-200, 200))),
                                   name="hue"),
                iaa.WithColorspace(to_colorspace="HSV",
                                   from_colorspace="RGB",
                                   children=iaa.WithChannels(
                                       1, iaa.Add((-20, 20))),
                                   name="sat"),
                iaa.contrast.LinearContrast(
                    (0.5, 1.5), per_channel=0.2, name="norm"),
                iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
            ])),

        # Blur and Noise
        iaa.Sometimes(
            0.2,
            iaa.SomeOf((1, None), [
                iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                           iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
                iaa.OneOf([
                    iaa.AddElementwise(
                        (-5, 5), per_channel=0.5, name="add-element"),
                    iaa.MultiplyElementwise(
                        (0.95, 1.05), per_channel=0.5, name="mul-element"),
                    iaa.AdditiveGaussianNoise(
                        scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                    iaa.AdditiveLaplaceNoise(
                        scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                    iaa.Sometimes(1.0, iaa.Dropout(
                        p=(0.003, 0.01), per_channel=0.5, name="dropout")),
                ]),
            ],
                random_order=True))
    ])

    # list of imgaug Transform names to be applied only on the imgs.
    input_only = [
        "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
        "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
    ]

    # trainig data set
    db_synthetic_lst = []
    if config.train.datasetsTrain is not None:
        for dataset in config.train.datasetsTrain:
            if dataset.images:
                db_synthetic = dataloader.ClearGraspsDataset(input_dir=dataset.images,
                                                             depth_dir=dataset.depths,
                                                             transform=augs_train,
                                                             input_only=input_only)
                                                        
                train_size = int(
                    config.train.percentageDataForTraining * len(db_synthetic))
                db_synthetic = torch.utils.data.Subset(db_synthetic, range(train_size))
                db_synthetic_lst.append(db_synthetic)
        db_synthetic = torch.utils.data.ConcatDataset(db_synthetic_lst)

    # Validation data set
    augs_test = iaa.Sequential([
        iaa.Resize({
            "height": config.train.imgHeight,
            "width": config.train.imgWidth
        }, interpolation='nearest'),
    ])

    db_val_list_real = []
    if config.train.datasetsValReal is not None:
        for dataset in config.train.datasetsValReal:
            if dataset.images:
                db = dataloader.ClearGraspsDataset(input_dir=dataset.images,
                                                   depth_dir=dataset.depths,
                                                   transform=augs_test,
                                                   input_only=None,
                                                   isSynthetic=False)

                train_size = int(
                    config.train.percentageDataForValidation * len(db))
                db = torch.utils.data.Subset(db, range(train_size))
                db_val_list_real.append(db)

    if db_val_list_real:
        db_val_real = torch.utils.data.ConcatDataset(db_val_list_real)

    db_val_synthetic_list = []
    if config.train.datasetsValSynthetic is not None:
        for dataset in config.train.datasetsValSynthetic:
            if dataset.images:
                db = dataloader.ClearGraspsDataset(input_dir=dataset.images,
                                                   depth_dir=dataset.depths,
                                                   transform=augs_test,
                                                   input_only=None)
                train_size = int(
                    config.train.percentageDataForValidation * len(db))
                db = torch.utils.data.Subset(db, range(train_size))
                db_val_synthetic_list.append(db)

    if db_val_synthetic_list:
        db_val_synthetic = torch.utils.data.ConcatDataset(
            db_val_synthetic_list)

    # Create dataloaders
    if db_synthetic_lst:
        assert (config.train.batchSize <= len(db_synthetic)), \
            ('batchSize ({}) cannot be more than the ' +
             'number of images in train dataset: {}').format(config.train.validationBatchSize, len(db_synthetic))

        trainLoader = DataLoader(db_synthetic,
                                 batch_size=config.train.batchSize,
                                 shuffle=True,
                                 num_workers=config.train.numWorkers,
                                 drop_last=True,
                                 pin_memory=True)

    if db_val_list_real:
        assert (config.train.validationBatchSize <= len(db_val_real)), \
            ('validationBatchSize ({}) cannot be more than the ' +
             'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val_real))

        realValidationLoader = DataLoader(db_val_real,
                                          batch_size=config.train.validationBatchSize,
                                          shuffle=True,
                                          num_workers=config.train.numWorkers,
                                          drop_last=False)

    # Create dataloaders
    if db_val_synthetic_list:
        assert (config.train.validationBatchSize <= len(db_val_synthetic)), \
            ('validationBatchSize ({}) cannot be more than the ' +
             'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val_synthetic))

        syntheticValidationLoader = DataLoader(db_val_synthetic,
                                               batch_size=config.train.validationBatchSize,
                                               shuffle=True,
                                               num_workers=config.train.numWorkers,
                                               drop_last=False)
    return trainLoader, syntheticValidationLoader, realValidationLoader

def loadClearGraspDataSet(config):
    '''Returns the DataLoader classes for training and validation data sets with respect to specified arguments 
        in the config yaml file. 

        Args:
            config (dict): The parsed configuration YAML file

        Returns:
            DataLoader: DataLoader for trainig data set
            DataLoader: DataLoader for synthetic validation data set
            DataLoader: DataLoader for real validation data set
    '''

    # trainig data set
    db_synthetic_lst = []
    if config.train.datasetsTrain is not None:
        for dataset in config.train.datasetsTrain:
            if dataset.images:
                db_synthetic = dataloader_clearGrasp.ClearGraspsDataset(input_dir=dataset.images,
                                                             depth_dir=dataset.depths, 
                                                             width=config.train.imgWidth,
                                                             height=config.train.imgHeight)
                                                        
                train_size = int(
                    config.train.percentageDataForTraining * len(db_synthetic))
                db_synthetic = torch.utils.data.Subset(db_synthetic, range(train_size))
                db_synthetic_lst.append(db_synthetic)
        db_synthetic = torch.utils.data.ConcatDataset(db_synthetic_lst)


    db_val_list_real = []
    if config.train.datasetsValReal is not None:
        for dataset in config.train.datasetsValReal:
            if dataset.images:
                db = dataloader_clearGrasp.ClearGraspsDataset(input_dir=dataset.images,
                                                             depth_dir=dataset.depths, 
                                                             width=config.train.imgWidth,
                                                             height=config.train.imgHeight,
                                                             mode='test',
                                                             isSynthetic=False)

                train_size = int(
                    config.train.percentageDataForValidation * len(db))
                db = torch.utils.data.Subset(db, range(train_size))
                db_val_list_real.append(db)

    if db_val_list_real:
        db_val_real = torch.utils.data.ConcatDataset(db_val_list_real)

    db_val_synthetic_list = []
    if config.train.datasetsValSynthetic is not None:
        for dataset in config.train.datasetsValSynthetic:
            if dataset.images:
                db = dataloader_clearGrasp.ClearGraspsDataset(input_dir=dataset.images,
                                                             depth_dir=dataset.depths, 
                                                             width=config.train.imgWidth,
                                                             height=config.train.imgHeight,
                                                             mode='test')
                train_size = int(
                    config.train.percentageDataForValidation * len(db))
                db = torch.utils.data.Subset(db, range(train_size))
                db_val_synthetic_list.append(db)

    if db_val_synthetic_list:
        db_val_synthetic = torch.utils.data.ConcatDataset(
            db_val_synthetic_list)

    # Create dataloaders
    if db_synthetic_lst:
        assert (config.train.batchSize <= len(db_synthetic)), \
            ('batchSize ({}) cannot be more than the ' +
             'number of images in train dataset: {}').format(config.train.validationBatchSize, len(db_synthetic))

        trainLoader = DataLoader(db_synthetic,
                                 batch_size=config.train.batchSize,
                                 shuffle=True,
                                 num_workers=config.train.numWorkers,
                                 drop_last=True,
                                 pin_memory=True)

    if db_val_list_real:
        assert (config.train.validationBatchSize <= len(db_val_real)), \
            ('validationBatchSize ({}) cannot be more than the ' +
             'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val_real))

        realValidationLoader = DataLoader(db_val_real,
                                          batch_size=config.train.validationBatchSize,
                                          shuffle=True,
                                          num_workers=config.train.numWorkers,
                                          drop_last=False)

    # Create dataloaders
    if db_val_synthetic_list:
        assert (config.train.validationBatchSize <= len(db_val_synthetic)), \
            ('validationBatchSize ({}) cannot be more than the ' +
             'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val_synthetic))

        syntheticValidationLoader = DataLoader(db_val_synthetic,
                                               batch_size=config.train.validationBatchSize,
                                               shuffle=True,
                                               num_workers=config.train.numWorkers,
                                               drop_last=False)
    return trainLoader, syntheticValidationLoader, realValidationLoader

def loadtransDepthDataSet(config):
    validationLoader = dataloader_transDepth.TransDepthDataLoader(config.train.transDepthDatasetVal, config.train.batchSize, config.train.numWorkers, 'val').data

    trainLoader = dataloader_transDepth.TransDepthDataLoader(config.train.transDepthDatasetTrain, config.train.batchSize, config.train.numWorkers, 'train').data

    return trainLoader, validationLoader 

def train(gpu, config, writer):
    '''Initializes the model, optimizer, and lr_scheduler specified in config. The model is moved 
        to provided gpu's. After setup, the training procedur of corresponding model is called.

        Args:
            gpu (list of int): Device IDs of GPUs which are going to be used
            config (dict): The parsed configuration YAML file
            writer (SummaryWriter): A Tensorboard SummaryWriter instance

        Raises:
            ValueError: If the given model is not supported
            ValueError: If the given loss function is not supported
            ValueError: If the given lr scheduler is not supported
    '''

    ###################### DataLoader #############################
    # Train Dataset - Create a dataset object for each dataset in our list, Concatenate datasets, select subset for training
    if config.train.dataset == 'transDepth':
        trainLoader, syntheticValidationLoader = loadtransDepthDataSet(config)
        realValidationLoader = None
    else:
        trainLoader, syntheticValidationLoader, realValidationLoader = loadClearGraspDataSet(config)

    ###################### ModelBuilder #############################
    if config.train.model == 'densedepth':
        from models.DenseDepth import DenseDepth
        model = DenseDepth()
    elif config.train.model == 'adabin':
        from models.AdaBin import UnetAdaptiveBins
        model = UnetAdaptiveBins.build(n_bins=256, min_val=config.train.min_depth,
                                       max_val=config.train.max_depth, norm='linear')
    elif config.train.model == 'dpt':
        from models.DPT.models import DPTDepthModel
        model = DPTDepthModel(
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        ) 
    elif config.train.model == 'lapdepth':
        from models.LapDepth import LDRN
        model = LDRN({
                'lv6': False,
                'encoder': 'ResNext101',
                'norm': 'BN',
                'act': 'ReLU',
                'max_depth': config.train.max_depth,
            })
    elif config.train.model == 'newcrf':
        from models.NewCRFDepth.NewCRFDepth import NewCRFDepth 
        model = NewCRFDepth(version='large07', inv_depth=False, max_depth=config.train.max_depth, pretrained=None)
    elif config.train.model == 'depthformer':
        from depth.models import build_depther
        import mmcv
        cfg = mmcv.Config.fromfile(config.train.depthformer.modelPath)
        model = build_depther(
        cfg.model,
        train_cfg=None,
        test_cfg=None)
        #model.init_weights()
        del cfg
    elif config.train.model == 'binsformer':
        from depth.models import build_depther
        import mmcv
        cfg = mmcv.Config.fromfile(config.train.binsformer.modelPath)
        model = build_depther(
        cfg.model,
        train_cfg=None,
        test_cfg=None)
        #model.init_weights()
        del cfg
    elif config.train.model == 'glp':
        from models.GLPDepth.model import GLPDepth
        model = GLPDepth(max_depth=config.train.max_depth, is_train=False)
    else:
        raise ValueError(
            'Invalid model "{}" in config file. Must be one of ["densedepth", "adabin", "dpt"]'
            .format(config.train.model))

    if config.train.continueTraining:
        if config.train.loadProjektCheckpoints:
            CHECKPOINT, model = model_io.load_checkpoint(config.train.pathPrevCheckpoint, model)
        else:
            model = model_io.load_origin_Checkpoint(config.train.pathPrevCheckpoint, config.train.model, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable Multi-GPU training
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, gpu)
    model = model.to(device)

    ###################### Setup Optimizer #############################
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(
                                      config.train.optimAdamW.learningRate),
                                  weight_decay=float(config.train.optimAdamW.weightDecay))

    # Continue Training from prev checkpoint if required
    if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
        if 'optimizer_state_dict' in CHECKPOINT:
            optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
            #optimizer.step(START_EPOCH)
        else:
            print(
                colored(
                    'WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                    '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))


    if config.train.lossFunc == 'SSIM':    
        criterion = loss_functions.ssim
    elif config.train.lossFunc == 'SILog':
        criterion = loss_functions.SILogLoss()
    else:
        raise ValueError(
            "Invalid Scheduler from config file: '{}'. Valid values are ['SSIM','SILog']".format(
                config.train.lossFunc))
    

    if config.train.lrScheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  factor=float(
                                                                      config.train.lrSchedulerPlateau.factor),
                                                                  patience=config.train.lrSchedulerPlateau.patience,
                                                                  verbose=True)
    elif config.train.lrScheduler == 'OneCycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(config.train.optimAdamW.learningRate),
                                                           epochs=config.train.numEpochs, steps_per_epoch=len(
                                                               trainLoader),
                                                           cycle_momentum=True,
                                                           base_momentum=0.85, max_momentum=0.95,
                                                           div_factor=config.train.OneCycleLR.div_factor,
                                                           final_div_factor=config.train.OneCycleLR.final_div_factor)
    else:
        raise ValueError(
            "Invalid Scheduler from config file: '{}'. Valid values are ['ReduceLROnPlateau, 'OneCycleLR']".format(
                config.train.lrScheduler))

    ###################### Train Model #############################
    # Set total iter_num (number of batches seen by model, used for logging)
    total_iter_num = 0
    START_EPOCH = 0
    END_EPOCH = config.train.numEpochs

    if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
        total_iter_num = CHECKPOINT['total_iter_num'] + 1
        START_EPOCH = CHECKPOINT['epoch'] 
        if CHECKPOINT['epoch'] == config.train.numEpochs:
            END_EPOCH = config.train.numEpochs + 1

        #if config.train.lrScheduler == 'OneCycleLR':
        #    lr_scheduler.step(START_EPOCH)

    framework_train.train(config.train.model, writer, device, model, trainLoader, syntheticValidationLoader, realValidationLoader, optimizer,criterion,  lr_scheduler, START_EPOCH,
                                  END_EPOCH, total_iter_num, config.train.validateModelInterval, CHECKPOINT_DIR, config_yaml, config)
    


if __name__ == '__main__':

    ###################### Load Config File #############################
    parser = argparse.ArgumentParser(
        description='Run training of outlines prediction model')
    parser.add_argument('-c', '--configFile', required=True,
                        help='Path to config yaml file', metavar='path/to/config')
    args = parser.parse_args()

    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        # Returns an ordered dict. Used for printing
        config_yaml = oyaml.load(fd, Loader=oyaml.FullLoader)

    config = AttrDict(config_yaml)
    # print(colored('Config being used for training:\n{}\n\n'.format(oyaml.dump(config_yaml)), 'green'))

    ###################### Logs (TensorBoard)  #############################
    # Create directory to save results
    SUBDIR_RESULT = 'checkpoints'

    results_root_dir = config.train.logsDir
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
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    MODEL_LOG_DIR = results_dir
    CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, SUBDIR_RESULT)
    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('Saving results to folder: ' +
          colored('"{}"'.format(results_dir), 'blue'))

    # Create a tensorboard object and Write config to tensorboard
    writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

    string_out = io.StringIO()
    oyaml.dump(config_yaml, string_out, default_flow_style=False)
    config_str = string_out.getvalue().split('\n')
    string = ''
    for line in config_str:
        string = string + '    ' + line + '\n\r'
    writer.add_text('Config', string, global_step=None)

    train(device_ids, config, writer)

    writer.close()
