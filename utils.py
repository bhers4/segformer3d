import numpy as np
import os
import torch

def generate_fake_data(num_samples: int, output_dir: str):
    """
        Since we aren't allowed to share the actual data used in this paper,
        we will generate some fake data so one can run the code and see the
        dimensions of different blocks.
    """
    print("Input dir: ", output_dir)
    training_dir = output_dir + "/training"
    validation_dir = output_dir + "/validation"
    testing_dir = output_dir + "/testing"
    if not os.path.isdir(training_dir):
        os.makedirs(training_dir)
    if not os.path.isdir(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.isdir(testing_dir):
        os.makedirs(testing_dir)
    # Our volumes we geometrically sampled and then downsampled to 100^3.
    fake_training_files = []
    fake_validation_files = []
    fake_testing_files = []
    for dir in [training_dir, validation_dir, testing_dir]:
        for i in range(num_samples):
            vol = np.random.rand(100, 100, 100)
            if not os.path.isfile(f"{dir}/vol_{i}.npy"):
                np.save(f"{dir}/vol_{i}.npy", vol)
            if dir == training_dir:
                fake_training_files.append(f"{dir}/vol_{i}.npy")
            elif dir == validation_dir:
                fake_validation_files.append(f"{dir}/vol_{i}.npy")
            else:
                fake_testing_files.append(f"{dir}/vol_{i}.npy")
    return fake_training_files, fake_validation_files, fake_testing_files

def find_files(directory: str, extension: str=".npy"):
    """
        If we are using the actual data, we need to find the files in the
        directory. We assume that the volumes and labels are stored in the
        same directory and that they have the same name except for the
        identifier. We converted volumes from .img to .npy files to make
        the training code simpler.
    """
    # Find raw volumes
    volume_identifier = "participant"
    # Find labels
    label_identifier = "Segmentation"
    volumes = []
    labels = []
    for root, dirs, files in os.walk(directory):
        # Volume and label files are stored in same directory so find files
        # and only add them if they contain the volume and label.
        vol = None
        label = None
        for file in files:
            if file.endswith(extension):
                if volume_identifier in file:
                    vol = os.path.join(root, file)
                elif label_identifier in file:
                    label = os.path.join(root, file)
        if vol is not None and label is not None:
            volumes.append(vol)
            labels.append(label)
    print("Found: ", len(volumes), " volumes and ", len(labels), " labels.")
    return volumes, labels