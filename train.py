#!/usr/bin/ python3
# Pip imports
import argparse
from scipy import ndimage
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Union
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import time
# Custom imports
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

def main(args):
    input_dir = args.input_dir
    
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
    model = SegFormer(zoom_size=128, patch_size=7, num_classes=1)
    model.cuda()

    # Create the optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Create loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Set number of epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        # Start timing the epoch
        epoch_start_time = time.time()
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for i, (volume, label) in enumerate(training_dataloader):
            optimizer.zero_grad()
            volume = volume.unsqueeze(1).float().cuda()
            label = label.unsqueeze(1).float().cuda()
            output = model(volume)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            training_iter = epoch * len(training_dataloader) + i
            if training_iter % 100 == 0:
                vol_img = volume[0, 0, :, :, 64].cpu().detach().numpy()
                writer.add_image('Volume', vol_img[None], training_iter)
                label_img = label[0, 0, :, :, 64].cpu().detach().numpy()
                writer.add_image('Label', label_img[None], training_iter)
                output_img = output[0, 0, :, :, 64].cpu().detach().numpy()
                writer.add_image('Output', output_img[None], training_iter)

            # Log batch loss
            writer.add_scalar('Loss/train_batch', loss.item(),
                              epoch * len(training_dataloader) + i)
            
            # Accumulate loss for epoch average
            epoch_loss += loss.item()
            num_batches += 1

            print("Epoch: ", epoch, "Batch: ", i, "Loss: ", loss.item())
        
        # Calculate and log average training loss for the epoch
        avg_train_loss = epoch_loss / num_batches
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        
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
        
        # Print epoch results including duration
        print(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.2f} seconds")
    
    # Close the tensorboard writer
    writer.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--generate-fake-data", action="store_true",
                        help="Generate fake data for training/testing purposes.")
    args = parser.parse_args()
    # Do something with the arguments
    print(torch.__version__)
    main(args)
