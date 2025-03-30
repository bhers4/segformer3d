#!/usr/bin/ python3
# Pip imports
import argparse
from datetime import datetime
import json
import numpy as np
import os
from scipy import ndimage
import time
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Union
from torch.utils.tensorboard import SummaryWriter
# Custom imports
from loss import DiceBCELoss
from utils import generate_fake_data, find_files
from segformer3d import SegFormer

class UltrasoundDataset(Dataset):
    def __init__(self, volumes, labels):
        self.volumes = volumes
        self.labels = labels
        self.shape_0_min = 210
        self.shape_1_min = 200
        self.shape_2_min = 160

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume = np.load(self.volumes[idx])
        # Geometrically sample since each volume has the same pixel spacing but
        # different dimensions need to be sampled to the same size then resized.
        shape_00, shape_01, shape_10, shape_11, shape_20, shape_21 = \
            UltrasoundDataset.random_geo_crop(volume[None], self.shape_0_min,
                                              self.shape_1_min, self.shape_2_min)
        label = np.load(self.labels[idx])
        volume = volume[shape_00:shape_01, shape_10:shape_11, shape_20:shape_21]
        label = label[shape_00:shape_01, shape_10:shape_11, shape_20:shape_21]
        volume = UltrasoundDataset.zoom_img(volume, 128)
        label = UltrasoundDataset.zoom_img(label, 128)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume[0, :, :, :], label[0, :, :, :]

    @staticmethod
    def random_geo_crop(label: torch.Tensor, shape_0_min: int, shape_1_min: int,
                        shape_2_min: int):
        s0_range = np.abs(label.shape[1] - shape_0_min)
        s1_range = np.abs(label.shape[2] - shape_1_min)
        s2_range = np.abs(label.shape[3] - shape_2_min)

        rand_0 = np.random.randint(low=0, high=s0_range - 1, size=(1,))
        rand_1 = np.random.randint(low=0, high=s1_range - 1, size=(1,))
        rand_2 = np.random.randint(low=0, high=s2_range - 1, size=(1,))

        shape_00 = int(rand_0[0])
        shape_01 = int(rand_0[0]) + shape_0_min
        shape_10 = int(rand_1[0])
        shape_11 = int(rand_1[0]) + shape_1_min
        shape_20 = int(rand_2[0])
        shape_21 = int(rand_2[0]) + shape_2_min
        return shape_00, shape_01, shape_10, shape_11, shape_20, shape_21
    
    @staticmethod
    def zoom_img(img: Union[np.ndarray, torch.Tensor], zoom_size: int) -> torch.Tensor:
        if len(img.shape) == 3:
            x_zoom = zoom_size / img.shape[0]
            y_zoom = zoom_size / img.shape[1]
            z_zoom = zoom_size / img.shape[2]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
        else:
            x_zoom = zoom_size / img.shape[1]
            y_zoom = zoom_size / img.shape[2]
            z_zoom = zoom_size / img.shape[3]
            img = img[0,:,:,:].numpy()
        img = ndimage.zoom(img, zoom=(x_zoom, y_zoom, z_zoom))
        img = torch.tensor(img[None,:,:,:], dtype=torch.float)
        return img

def calculate_dice_and_iou(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    thresholded = pred.clone().detach()
    thresholded[thresholded > 0.5] = 1.0
    thresholded[thresholded != 1.0] = 0.0
    label = target.clone().detach()
    label[label > 0.5] = 1.0
    label[label <= 0.5] = 0.0
    confusion_vector = thresholded / label
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    if (2*true_positives + false_positives + false_negatives) == 0:
        dice = 0.0
    else:
        dice = 2*true_positives / (2*true_positives + false_positives + false_negatives)
    iou_and = (torch.logical_and(thresholded, label)).sum()
    iou_or = (torch.logical_or(thresholded, label)).sum()
    iou = iou_and / iou_or
    return dice, iou

def main(args):
    input_dir = args.input_dir
    # Load the json config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    # Create tensorboard log directory
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    if args.generate_fake_data:
        training_data, validation_data, testing_data = \
            generate_fake_data(num_samples=100, output_dir=input_dir)
        # This is fake data so doesn't matter what the labels are.
        training_labels = training_data
        validation_labels = validation_data
        testing_labels = testing_data
    else:
        volumes, labels = find_files(directory=input_dir, extension=".npy")
        """
            In Segformer3d paper, we split data by patient as there was multiple volumes per patient but in this demo we
            show a simple training loop. For other medical imaging applications close attention should be paid to not
            having data leakage between training, validation and test sets.
        """
        # Generate a train, validation, and test split
        training_data = volumes[:int(0.6*len(volumes))]
        validation_data = volumes[int(0.6*len(volumes)):int(0.8*len(volumes))]
        testing_data = volumes[int(0.8*len(volumes)):]
        training_labels = labels[:int(0.6*len(labels))]
        validation_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]
        testing_labels = labels[int(0.8*len(labels)):]
    
    # Create the datasets
    training_dataset = UltrasoundDataset(training_data, training_labels)
    validation_dataset = UltrasoundDataset(validation_data, validation_labels)
    testing_dataset = UltrasoundDataset(testing_data, testing_labels)

    # Create the dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

    # Create the model
    model = SegFormer(zoom_size=config['model']['zoom_size'], patch_size=config['model']['patch_size'], num_classes=1)
    model.cuda()

    # Create the optimizer
    if config['optimizer']['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['params']['lr'],
                                     weight_decay=config['optimizer']['params']['weight_decay'],
                                     amsgrad=config['optimizer']['params']['amsgrad'])
    elif config['optimizer']['name'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['params']['lr'],
                                      weight_decay=config['optimizer']['params']['weight_decay'],
                                      amsgrad=config['optimizer']['params']['amsgrad'])
    else:
        raise ValueError("Optimizer not supported")
    # Create the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler']['params']['step_size'],
                                                   gamma=config['scheduler']['params']['gamma'])
    # Create loss function, default to BCEWithLogitsLoss
    if config['loss'] == 'DiceBCELoss':
        loss_fn = DiceBCELoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    # Log message to tensorboard
    writer.add_text('Message', args.message)
    # Log the json config to tensorboard so we can correlate the config used with each run.
    writer.add_text('Config', json.dumps(config, indent=4))
    # Set number of epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        # Start timing the epoch
        epoch_start_time = time.time()
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        # Dice and IoU metrics
        dice_epoch = 0.0
        iou_epoch = 0.0
        # Iterate over the training data
        print("Starting epoch %d" % epoch)
        for i, (volume, label) in enumerate(training_dataloader):
            optimizer.zero_grad()
            volume = volume.unsqueeze(1).float().cuda()
            label = label.unsqueeze(1).float().cuda()
            output = model(volume)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            # Log slices of the volume, label and output to see the model behavior
            training_iter = epoch * len(training_dataloader) + i
            if training_iter % 100 == 0:
                # Find slice with most label/information in dimension 2
                label_img = label[0, 0, :, :, :].cpu().detach().numpy()
                label_sum = np.sum(label_img, axis=(0, 1))
                slice_index = np.argmax(label_sum)
                vol_img = volume[0, 0, :, :, slice_index].cpu().detach().numpy()
                writer.add_image('Volume', vol_img[None], training_iter)
                label_img = label[0, 0, :, :, slice_index].cpu().detach().numpy()
                writer.add_image('Label', label_img[None], training_iter)
                output_img = output[0, 0, :, :, slice_index].cpu().detach().numpy()
                writer.add_image('Output', output_img[None], training_iter)
            # Print the loss for the current batch on the same line
            print(f"Epoch {epoch}, Batch {i + 1}/{len(training_dataloader)}, Loss: {loss.item():.4f}", end='\r')
            # Log batch loss
            writer.add_scalar('Loss/train_batch', loss.item(),
                              epoch * len(training_dataloader) + i)
            # Calculate Dice and IoU and log them
            dice, iou = calculate_dice_and_iou(output, label)
            dice_epoch += dice
            iou_epoch += iou
            writer.add_scalar('Dice/train_batch', dice, epoch * len(training_dataloader) + i)
            writer.add_scalar('IoU/train_batch', iou, epoch * len(training_dataloader) + i)
            # Accumulate loss for epoch average
            epoch_loss += loss.item()
            num_batches += 1
        print()
        # Update scheduler
        lr_scheduler.step()
        # Log learning rate
        writer.add_scalar('LearningRate/epoch', optimizer.param_groups[0]['lr'], epoch)
        # Calculate and log average training loss for the epoch
        avg_train_loss = epoch_loss / num_batches
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        # Calculate and log average Dice and IoU for the epoch
        avg_dice = dice_epoch / num_batches
        avg_iou = iou_epoch / num_batches
        writer.add_scalar('Dice/train_epoch', avg_dice, epoch)
        writer.add_scalar('IoU/train_epoch', avg_iou, epoch)
        # Run validation set for each epoch
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for i, (volume, label) in enumerate(validation_dataloader):
                volume = volume.unsqueeze(1).float().cuda()
                label = label.unsqueeze(1).float().cuda()
                output = model(volume)
                loss = loss_fn(output, label)
                # Accumulate validation loss
                val_loss += loss.item()
                val_batches += 1
        # Calculate and log average validation loss for the epoch
        avg_val_loss = val_loss / val_batches
        writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        # Log epoch duration
        writer.add_scalar('Time/epoch_duration_seconds', epoch_duration, epoch)
        # Calculate time left in training
        time_left = (num_epochs - epoch) * epoch_duration
        # Print epoch results including duration
        print(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.2f} seconds, Time left: {time_left:.2f} seconds")
    
    # Close the tensorboard writer
    writer.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file.")
    parser.add_argument("--generate-fake-data", action="store_true",
                        help="Generate fake data for training/testing purposes.")
    parser.add_argument("--message", type=str, default="", 
                        help="Message to log to tensorboard.")
    args = parser.parse_args()
    # Do something with the arguments
    print(torch.__version__)
    main(args)
