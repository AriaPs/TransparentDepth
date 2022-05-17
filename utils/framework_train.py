
'''

Training framework for depth map estimation of transparent structures.

Note: This file is adapted from ClearGrasp trainig framework.

'''

from utils.model_io import save_model
from utils.api import create_grid_image
import loss_functions
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from imgaug import augmenters as iaa
import torch.nn as nn
import torch
import numpy as np
import os


def trainDenseDepth(writer, device, model, trainLoader, syntheticValidationLoader, realValidationLoader, optimizer, criterion, lr_scheduler, START_EPOCH,
                    END_EPOCH, total_iter_num, validateModelInterval, CHECKPOINT_DIR, config_yaml):
    '''Trains the [model] for n = [END_EPOCH] - [START_EPOCH] epochs on the DataSet loaded by [trainLoader]. In each iteration, 
    a forward is taken, loss is computed with [criterion], and backward pass is taken. Finally, [optimizer] and [lr_scheduler] take a step.
    Moreover, the model is validated [validateModelInterval] times and a checkpoint is saved if the loss is the best. A checkpoint is also saved after each epoch. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            trainLoader (DataLoader): DataLoader for trainig
            syntheticValidationLoader (DataLoader): DataLoader for synthetic validation set
            realValidationLoader (DataLoader): DataLoader for real validation set
            optimizer (torch.optim.AdamW): Instance of optimizer
            criterion (Function): The loss function
            lr_scheduler (torch.optim.lr_scheduler): Instance of lr scheduler
            START_EPOCH (int): Firs epoch
            END_EPOCH (int): Last epoch 
            total_iter_num (int): The total number of iteration the model has seen
            validateModelInterval (int): Specify how many time the model should be validated in a epoch
            CHECKPOINT_DIR (str): The path where the checkpoint should be saved
            config_yaml (dict): The parsed configuration YAML file

    '''

    best_loss = np.inf
    num_samples = (len(trainLoader))
    validateInterval = np.ceil(num_samples / validateModelInterval)
    for epoch in range(START_EPOCH, END_EPOCH):
        print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
        print('-' * 30)

        # Log the current Epoch Number
        writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

        ###################### Training Cycle #############################
        print('Train:')
        print('=' * 10)

        model.train()

        running_loss = 0.0
        for iter_num, batch in enumerate(tqdm(trainLoader)):
            total_iter_num += 1

            # Get data
            input_norm, depths = batch

            #  Forward + Backward Prop
            optimizer.zero_grad()
            torch.set_grad_enabled(True)

            model_output = model(input_norm.to(device))

            # Compute the loss
            loss = criterion(model_output, depths.to(device))

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            writer.add_scalar('data/Train BatchWise Loss',
                              loss.item(), total_iter_num)
            
            lr_scheduler.step()
            
            
            current_learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar(
                'Learning Rate', current_learning_rate, total_iter_num)

            if (iter_num % validateInterval) == 0:
                compare_loss = valDenseDepth(writer, device, model, syntheticValidationLoader,  criterion,  total_iter_num,
                            '1/Validation-synthetic-images-{}'.format(iter_num), 'synthetic')
                if realValidationLoader != None:
                    real_loss = valDenseDepth(writer, device, model, realValidationLoader,  criterion,  total_iter_num,
                                '2/Validation-real-images-{}'.format(iter_num), 'real')
                    compare_loss = (compare_loss + real_loss) / 2
                model.train()
                if(best_loss > compare_loss):
                    best_loss = compare_loss
                    save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                           total_iter_num, compare_loss, config_yaml, 'checkpoint-best.pth')
            

        # Log Epoch Loss
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
        print('Train Epoch Loss: {:.4f}'.format(epoch_loss))

        # Log images every epochs
        
        output = resize_tensor(model_output.detach().cpu(), int(model_output.shape[2] * 2),
                               int(model_output.shape[3] * 2))

        grid_image = create_grid_image(input_norm.detach(),
                                                    output,
                                                    depths.detach())
        writer.add_image('0/Train-images-{}'.format(epoch),
                         grid_image, total_iter_num)

        # Save the model checkpoint every epochs
        
        save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                   total_iter_num, epoch_loss, config_yaml, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
                
def valDenseDepth(writer, device, model, validationLoader,  criterion,  total_iter_num, writerTextInput, setTyp):
    '''Validates the [model] on the DataSet [setTyp] loaded by [validationLoader]. In each iteration, 
    a forward is taken and loss is computed with [criterion]. Finally, a grid image is generated and saved in Tensorboard. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            validationLoader (DataLoader): DataLoader for validation set
            criterion (Function): The loss function
            total_iter_num (int): The total number of iteration the model has seen
            writerTextInput (str): The text which will be the title of grid image in Tensorboard
            setTyp (str): The data set type as String (real or synthetic)
        
        Returns:
            float: The epoch loss 

    '''

    print('\nValidation-{}:'.format(setTyp))
    print('=' * 10)

    model.eval()

    running_loss = 0.0
    for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
        input_norm, depths = sample_batched

        with torch.no_grad():
            model_output = model(input_norm.to(device))

        # Compute the loss
        loss = criterion(model_output, depths.to(device))

        running_loss += loss.item()

        # Log Epoch Loss
    num_samples = (len(validationLoader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Validation {} Epoch Loss'.format(setTyp),
                      epoch_loss, total_iter_num)
    print('Validation {} Epoch Loss: {:.4f}'.format(setTyp, epoch_loss))

    output = resize_tensor(model_output.detach().cpu(), int(model_output.shape[2] * 2),
                           int(model_output.shape[3] * 2))
    grid_image= create_grid_image(input_norm.detach(),
                                                output,
                                                depths.detach().cpu())
    writer.add_image(
        '{}'.format(writerTextInput), grid_image, total_iter_num)

    return epoch_loss

def trainAdaBin(writer, device, model, trainLoader, syntheticValidationLoader, realValidationLoader, optimizer, criterion, lr_scheduler, START_EPOCH,
                END_EPOCH, total_iter_num, validateModelInterval, CHECKPOINT_DIR, config_yaml):
    
    '''Trains the [model] for n = [END_EPOCH] - [START_EPOCH] epochs on the DataSet loaded by [trainLoader]. In each iteration, 
    a forward is taken, loss is computed with [criterion], and backward pass is taken. Finally, [optimizer] and [lr_scheduler] take a step.
    Moreover, the model is validated [validateModelInterval] times and a checkpoint is saved if the loss is the best. A checkpoint is also saved after each epoch. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            trainLoader (DataLoader): DataLoader for trainig
            syntheticValidationLoader (DataLoader): DataLoader for synthetic validation set
            realValidationLoader (DataLoader): DataLoader for real validation set
            optimizer (torch.optim.AdamW): Instance of optimizer
            criterion (Function): The loss function
            lr_scheduler (torch.optim.lr_scheduler): Instance of lr scheduler
            START_EPOCH (int): Firs epoch
            END_EPOCH (int): Last epoch 
            total_iter_num (int): The total number of iteration the model has seen
            validateModelInterval (int): Specify how many time the model should be validated in a epoch
            CHECKPOINT_DIR (str): The path where the checkpoint should be saved
            config_yaml (dict): The parsed configuration YAML file

    '''

    criterion_bins = loss_functions.BinsChamferLoss()
    best_loss = np.inf
    num_samples = (len(trainLoader))
    validateInterval = np.ceil(num_samples / validateModelInterval)
    for epoch in range(START_EPOCH, END_EPOCH):
        print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
        print('-' * 30)

        # Log the current Epoch Number
        writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

        ###################### Training Cycle #############################
        print('Train:')
        print('=' * 10)

        model.train()

        running_loss = 0.0
        for iter_num, batch in enumerate(tqdm(trainLoader)):
            total_iter_num += 1

            # Get data
            input_norm, depths = batch

            #  Forward + Backward Prop
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            bin_edges, model_output = model(input_norm.to(device))

            l_dense = criterion(model_output, depths.to(device))
            l_chamfer = criterion_bins(bin_edges, depths.to(device))
            loss = l_dense + 0.1 * l_chamfer

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # statistics
            running_loss += loss.item()
            writer.add_scalar('data/Train BatchWise Loss',
                              loss.item(), total_iter_num)

            lr_scheduler.step()

            current_learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar(
                'Learning Rate', current_learning_rate, total_iter_num)

            if (iter_num % validateInterval) == 0:
                compare_loss = valAdaBin(writer, device, model, syntheticValidationLoader, criterion,  total_iter_num,
                            '1/Validation-synthetic-images-{}'.format(iter_num), 'synthetic')
                if realValidationLoader != None:
                    real_loss = valAdaBin(writer, device, model, realValidationLoader, criterion,  total_iter_num,
                                '2/Validation-real-images-{}'.format(iter_num), 'real')
                    compare_loss = (compare_loss + real_loss) / 2
                model.train()
                if(best_loss > compare_loss):
                    best_loss = compare_loss
                    save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                           total_iter_num, compare_loss, config_yaml, 'checkpoint-best.pth')

        # Log Epoch Loss
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
        print('Train Epoch Loss: {:.4f}'.format(epoch_loss))
        
        # Log images every epochs
        
        output = resize_tensor(model_output.detach().cpu(), int(model_output.shape[2] * 2),
                               int(model_output.shape[3] * 2))
                               
        grid_image = create_grid_image(input_norm.detach(),
                                                    output,
                                                    depths.detach())
        writer.add_image('0/Train-images-{}'.format(epoch),
                         grid_image, total_iter_num)

        # Save the model checkpoint every epochs
        save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                   total_iter_num, epoch_loss, config_yaml, 'checkpoint-epoch-{:04d}.pth'.format(epoch))

def valAdaBin(writer, device, model, validationLoader, criterion,  total_iter_num, writerTextInput, setTyp):
    '''Validates the [model] on the DataSet [setTyp] loaded by [validationLoader]. In each iteration, 
    a forward is taken and loss is computed with [criterion]. Finally, a grid image is generated and saved in Tensorboard. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            validationLoader (DataLoader): DataLoader for validation set
            criterion (Function): The loss function
            total_iter_num (int): The total number of iteration the model has seen
            writerTextInput (str): The text which will be the title of grid image in Tensorboard
            setTyp (str): The data set type as String (real or synthetic)
        
        Returns:
            float: The epoch loss 

    '''

    print('\nValidation-{}:'.format(setTyp))
    print('=' * 10)

    model.eval()

    running_loss = 0.0
    for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
        input_norm, depths = sample_batched

        with torch.no_grad():
            _, model_output = model(input_norm.to(device))

        # Compute the loss
        loss = criterion(model_output, depths.to(device))

        running_loss += loss.item()

    # Log Epoch Loss
    num_samples = (len(validationLoader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Validation {} Epoch Loss'.format(setTyp),
                      epoch_loss, total_iter_num)
    print('Validation {} Epoch Loss: {:.4f}'.format(setTyp, epoch_loss))

    output = resize_tensor(model_output.detach().cpu(), int(model_output.shape[2] * 2),
                           int(model_output.shape[3] * 2))
    grid_image= create_grid_image(input_norm.detach(),
                                                output,
                                                depths.detach())
    writer.add_image(
        '{}'.format(writerTextInput), grid_image, total_iter_num)
    
    return epoch_loss

def trainDPT(writer, device, model, trainLoader, syntheticValidationLoader, realValidationLoader, optimizer, criterion, lr_scheduler, START_EPOCH,
                    END_EPOCH, total_iter_num, validateModelInterval, CHECKPOINT_DIR, config_yaml):

    '''Trains the [model] for n = [END_EPOCH] - [START_EPOCH] epochs on the DataSet loaded by [trainLoader]. In each iteration, 
    a forward is taken, loss is computed with [criterion], and backward pass is taken. Finally, [optimizer] and [lr_scheduler] take a step.
    Moreover, the model is validated [validateModelInterval] times and a checkpoint is saved if the loss is the best. A checkpoint is also saved after each epoch. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            trainLoader (DataLoader): DataLoader for trainig
            syntheticValidationLoader (DataLoader): DataLoader for synthetic validation set
            realValidationLoader (DataLoader): DataLoader for real validation set
            optimizer (torch.optim.AdamW): Instance of optimizer
            criterion (Function): The loss function
            lr_scheduler (torch.optim.lr_scheduler): Instance of lr scheduler
            START_EPOCH (int): Firs epoch
            END_EPOCH (int): Last epoch 
            total_iter_num (int): The total number of iteration the model has seen
            validateModelInterval (int): Specify how many time the model should be validated in a epoch
            CHECKPOINT_DIR (str): The path where the checkpoint should be saved
            config_yaml (dict): The parsed configuration YAML file

    '''
    
    best_loss = np.inf
    num_samples = (len(trainLoader))
    validateInterval = np.ceil(num_samples / validateModelInterval)
    for epoch in range(START_EPOCH, END_EPOCH):
        print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
        print('-' * 30)

        # Log the current Epoch Number
        writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

        ###################### Training Cycle #############################
        print('Train:')
        print('=' * 10)

        model.train()

        running_loss = 0.0
        for iter_num, batch in enumerate(tqdm(trainLoader)):
            total_iter_num += 1

            # Get data
            image, gt = batch

            #  Forward + Backward Prop
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            
            model_output = model(image.to(device))

            # [BXHXW] --> [BX1XHXW]
            model_output = torch.unsqueeze(model_output, 1) 
            

            # Compute the loss
            loss = criterion(model_output, gt.to(device), interpolate=False)


            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            writer.add_scalar('data/Train BatchWise Loss',
                              loss.item(), total_iter_num)
            
            lr_scheduler.step()
            
            
            current_learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar(
                'Learning Rate', current_learning_rate, total_iter_num)

            if (iter_num % validateInterval) == 0:
                compare_loss = valDPT(writer, device, model, syntheticValidationLoader,  criterion,  total_iter_num,
                            '1/Validation-synthetic-images-{}'.format(iter_num), 'synthetic')
                if realValidationLoader != None:
                    real_loss = valDPT(writer, device, model, realValidationLoader,  criterion,  total_iter_num,
                                '2/Validation-real-images-{}'.format(iter_num), 'real')
                    compare_loss = (compare_loss + real_loss) / 2
                model.train()
                if(best_loss > compare_loss):
                    best_loss = compare_loss
                    save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                           total_iter_num, compare_loss, config_yaml, 'checkpoint-best.pth')

            del model_output
            del gt
            

        # Log Epoch Loss
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
        print('Train Epoch Loss: {:.4f}'.format(epoch_loss))

        # Log images every epochs
    
        #grid_image = create_grid_image(image.detach(),
        #                                            model_output.detach().cpu(),
        #                                            gt.detach())
        #writer.add_image('0/Train-images-{}'.format(epoch),
        #                 grid_image, total_iter_num)

        # Save the model checkpoint every epochs
    
        save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                   total_iter_num, epoch_loss, config_yaml, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
                
def valDPT(writer, device, model, validationLoader,  criterion,  total_iter_num, writerTextInput, setTyp):
    '''Validates the [model] on the DataSet [setTyp] loaded by [validationLoader]. In each iteration, 
    a forward is taken and loss is computed with [criterion]. Finally, a grid image is generated and saved in Tensorboard. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            validationLoader (DataLoader): DataLoader for validation set
            criterion (Function): The loss function
            total_iter_num (int): The total number of iteration the model has seen
            writerTextInput (str): The text which will be the title of grid image in Tensorboard
            setTyp (str): The data set type as String (real or synthetic)
        
        Returns:
            float: The epoch loss 

    '''

    print('\nValidation-{}:'.format(setTyp))
    print('=' * 10)

    model.eval()

    running_loss = 0.0
    for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
        image, gt = sample_batched

        
        with torch.no_grad():
            model_output = model(image.to(device))
            model_output = torch.unsqueeze(model_output, 1)

        # Compute the loss
        loss = criterion(model_output, gt.to(device))

        running_loss += loss.item()

        # Log Epoch Loss
    num_samples = (len(validationLoader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Validation {} Epoch Loss'.format(setTyp),
                      epoch_loss, total_iter_num)
    print('Validation {} Epoch Loss: {:.4f}'.format(setTyp, epoch_loss))

    grid_image= create_grid_image(image.detach(),
                                                model_output.detach().cpu(),
                                                gt.detach().cpu())
    writer.add_image(
        '{}'.format(writerTextInput), grid_image, total_iter_num)
        
    return epoch_loss

def trainLapDepth(writer, device, model, trainLoader, syntheticValidationLoader, realValidationLoader, optimizer, criterion, lr_scheduler, START_EPOCH,
                END_EPOCH, total_iter_num, validateModelInterval, CHECKPOINT_DIR, config_yaml):
    
    '''Trains the [model] for n = [END_EPOCH] - [START_EPOCH] epochs on the DataSet loaded by [trainLoader]. In each iteration, 
    a forward is taken, loss is computed with [criterion], and backward pass is taken. Finally, [optimizer] and [lr_scheduler] take a step.
    Moreover, the model is validated [validateModelInterval] times and a checkpoint is saved if the loss is the best. A checkpoint is also saved after each epoch. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            trainLoader (DataLoader): DataLoader for trainig
            syntheticValidationLoader (DataLoader): DataLoader for synthetic validation set
            realValidationLoader (DataLoader): DataLoader for real validation set
            optimizer (torch.optim.AdamW): Instance of optimizer
            criterion (Function): The loss function
            lr_scheduler (torch.optim.lr_scheduler): Instance of lr scheduler
            START_EPOCH (int): Firs epoch
            END_EPOCH (int): Last epoch 
            total_iter_num (int): The total number of iteration the model has seen
            validateModelInterval (int): Specify how many time the model should be validated in a epoch
            CHECKPOINT_DIR (str): The path where the checkpoint should be saved
            config_yaml (dict): The parsed configuration YAML file

    '''

    best_loss = np.inf
    num_samples = (len(trainLoader))
    validateInterval = np.ceil(num_samples / validateModelInterval)
    for epoch in range(START_EPOCH, END_EPOCH):
        print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
        print('-' * 30)

        # Log the current Epoch Number
        writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

        ###################### Training Cycle #############################
        print('Train:')
        print('=' * 10)

        model.train()

        running_loss = 0.0
        for iter_num, batch in enumerate(tqdm(trainLoader)):
            total_iter_num += 1

            # Get data
            image, gt = batch

            #  Forward + Backward Prop
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            _ , model_output = model(image.to(device))

            loss = criterion(model_output, gt.to(device))

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            writer.add_scalar('data/Train BatchWise Loss',
                              loss.item(), total_iter_num)

            lr_scheduler.step()

            current_learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar(
                'Learning Rate', current_learning_rate, total_iter_num)

            if (iter_num % validateInterval) == 0:
                compare_loss = valLapDepth(writer, device, model, syntheticValidationLoader, criterion,  total_iter_num,
                            '1/Validation-synthetic-images-{}'.format(iter_num), 'synthetic')
                if realValidationLoader != None:
                    real_loss = valLapDepth(writer, device, model, realValidationLoader, criterion,  total_iter_num,
                                '2/Validation-real-images-{}'.format(iter_num), 'real')
                    compare_loss = (compare_loss + real_loss) / 2
                model.train()
                if(best_loss > compare_loss):
                    best_loss = compare_loss
                    save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                           total_iter_num, compare_loss, config_yaml, 'checkpoint-best.pth')
            del model_output
            del gt

        # Log Epoch Loss
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
        print('Train Epoch Loss: {:.4f}'.format(epoch_loss))
        
        # Log images every epochs
                               
        #grid_image = create_grid_image(image.detach(),
        #                                            model_output.detach().cpu(),
        #                                            gt.detach())
#
        #writer.add_image('0/Train-images-{}'.format(epoch),
        #                 grid_image, total_iter_num)

        # Save the model checkpoint every epochs
        save_model(CHECKPOINT_DIR, epoch, model, optimizer,
                   total_iter_num, epoch_loss, config_yaml, 'checkpoint-epoch-{:04d}.pth'.format(epoch))

def valLapDepth(writer, device, model, validationLoader, criterion,  total_iter_num, writerTextInput, setTyp):
    '''Validates the [model] on the DataSet [setTyp] loaded by [validationLoader]. In each iteration, 
    a forward is taken and loss is computed with [criterion]. Finally, a grid image is generated and saved in Tensorboard. 

        Args:
            writer (SummaryWriter): A Tensorboard SummaryWriter instance
            device (list of int): List of GPU ids on which the model is trained
            model (torch.nn.Module): The model which is going to be trained 
            validationLoader (DataLoader): DataLoader for validation set
            criterion (Function): The loss function
            total_iter_num (int): The total number of iteration the model has seen
            writerTextInput (str): The text which will be the title of grid image in Tensorboard
            setTyp (str): The data set type as String (real or synthetic)
        
        Returns:
            float: The epoch loss 

    '''

    print('\nValidation-{}:'.format(setTyp))
    print('=' * 10)

    model.eval()

    running_loss = 0.0
    for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
        image, gt = sample_batched

        with torch.no_grad():
            _, model_output = model(image.to(device))

        # Compute the loss
        loss = criterion(model_output, gt.to(device))

        running_loss += loss.item()

    # Log Epoch Loss
    num_samples = (len(validationLoader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Validation {} Epoch Loss'.format(setTyp),
                      epoch_loss, total_iter_num)
    print('Validation {} Epoch Loss: {:.4f}'.format(setTyp, epoch_loss))

    grid_image= create_grid_image(image.detach(),
                                                model_output.detach().cpu(),
                                                gt.detach())
    writer.add_image(
        '{}'.format(writerTextInput), grid_image, total_iter_num)
    
    return epoch_loss


def resize_tensor(input_tensor, height, width):
    '''Resizes the [input_tensor] to the given [height] and [width].

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

    return resized_array