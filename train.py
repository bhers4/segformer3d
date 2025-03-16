#!/usr/bin/ python3
# Pip imports
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# Custom imports
from utils import generate_fake_data, find_files

class UltrasoundDataset(Dataset):
    def __init__(self, volumes, labels):
        self.volumes = volumes
        self.labels = labels
    def __len__(self):
        return len(self.volumes)
    def __getitem__(self, idx):
        volume = np.load(self.volumes[idx])
        label = np.load(self.labels[idx])
        return volume, label

def main(args):
    input_dir = args.input_dir
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
    training_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
    testing_dataloader = DataLoader(testing_dataset, batch_size=4, shuffle=False)

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
