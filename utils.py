import numpy as np
import os
import scipy.ndimage as ndimage
import torch
from typing import Dict, List

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

## Data Transforms

def rotate_volume(vol, rot_x: int, rot_y: int, rot_z: int, order=3,
                  reshape=False) -> torch.Tensor:
    """ Uses scipy ndimage to do 3D volume rotations in x/y/z direction by N degrees """
    # If vol is a tensor unfortunately for scipy ndimage we need np arrays
    if isinstance(vol, torch.Tensor):
        vol = vol.numpy()
    # Check shape
    assert len(vol.shape) == 3, "Wrong input to rotate volume"
    
    vol = ndimage.rotate(
        vol,
        angle = rot_x,
        axes = (2,0),
        order = order,
        reshape=reshape
    )
    #yrot
    vol = ndimage.rotate(
        vol,
        angle = rot_y,
        axes = (2,1),
        order = order,
        reshape=reshape
    )
    #zrot
    vol = ndimage.rotate(
        vol,
        angle = rot_z, #degrees (positive is counter-clockwise)
        axes = (1,0),
        order = order,
        reshape=reshape
        )
    # Rescale volume to 0-1 because rotations can cause weird values
    if np.max(vol) != np.min(vol):
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    return torch.tensor(vol)

class Random90Flip(object):
    """ Randomly flips the volume/img/label plus or minus 90 degrees """
    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        img = sample['img']
        label = sample['label']
        assert len(img.shape) == 3, "We are expecting 3D not greater image or else the axises in the ndimage rotate will be off"
        assert len(label.shape) == 3, "We are expecting 3D not greater label or else the axises in the ndimage rotate will be off"
        # Randomly generate a number between 0 and 1 and flip depending on value
        # 0-0.2: rotate 90 degrees, 0.2-0.4: rotate -90 degrees
        flip_var = np.random.uniform(0, 1)
        if flip_var < 0.2:
            img = rotate_volume(img, 0, 0, 90, order=3)
            label = rotate_volume(label, 0, 0, 90, order=3)
        elif 0.2 < flip_var < 0.4:
            img = rotate_volume(img, 0, 0, -90, order=3)
            label = rotate_volume(label, 0, 0, -90, order=3)
        if len(label.shape) == 4:
            label = label[0, :, :, :]
        return {"img": img, "label": label}

class RandomBlur(object):
    """ Randomly blurs volume by gaussian blur with settable sigma """
    def __init__(self, sigma=3) -> None:
        self.sigma = sigma
        self.blur_prob = 0.5
        pass

    def __call__(self, sample: Dict):
        """ Gets img and label from Dict and applies filter 50% of the time """
        img = sample['img']
        label = sample['label']
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        blur_var = np.random.uniform(0, 1)
        if blur_var > self.blur_prob:
            img = ndimage.gaussian_filter(img, sigma=self.sigma)
            
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        img[img > 1] = 1.0
        img[img < 0] = 0.0
        return {"img": torch.tensor(img), "label": label}


class RandomNoise(object):
    """ Randomly applies normally distributed noise to volume with settable mean and std """
    def __init__(self, mean=0.05, std=0.01) -> None:
        self.mean = mean
        self.std = std
        self.noise_prob = 0.5
        pass

    def __call__(self, sample: Dict):
        img = sample['img']
        label = sample['label']
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        noise_var = np.random.uniform(0, 1)
        if noise_var > self.noise_prob:
            noise_vec = np.random.randn(*img.shape) * self.std + self.mean
            img += noise_vec
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        img[img > 1] = 1
        img[img < 0] = 0
        return {"img": torch.tensor(img), "label": label}


class RandomRotations(object):
    """ Does small rotations between min and max degree to do perturb the volumes in each of the axial directions """
    def __init__(self, min_degree=-10, max_degree=10) -> None:
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.rand_prob = 0.5  # If rand var [0, 1) is greater than this we apply random rotations
        pass

    def __call__(self, sample):
        img = sample['img']
        label = sample['label']
        rand_num = np.random.uniform(0, 1)
        # Sometimes want to show it unrotated versions
        if rand_num > self.rand_prob:
            x = np.random.randint(self.min_degree, self.max_degree)
            y = np.random.randint(self.min_degree, self.max_degree)
            z = np.random.randint(self.min_degree, self.max_degree)
            img = rotate_volume(img, x, y, z, order=3)
            label = rotate_volume(label, x, y, z, order=3)
        return {"img": img, "label": label}

class GammaDistortion(object):
    """ Does gamma distortion by raising all 0->1 values to certain power """
    def __init__(self, min_pow=0.8, max_pow=1.2) -> None:
        self.min_pow = min_pow
        self.max_pow = max_pow
        self.gamma_prob = 0.5  # If rand var [0, 1) is greater than this we apply gamma distortions
        pass

    def __call__(self, sample):
        img = sample['img']
        label = sample['label']
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        rand_num = np.random.uniform(0, 1)
        # Sometimes want to show it unrotated versions
        if rand_num > self.gamma_prob:
            gamma = np.random.uniform(self.min_pow, self.max_pow, 1)
            # Only change the img not the label
            img = np.power(img, gamma)
            img[img > 1] = 1
            img[img < 0] = 0
        # Check if label is torch tensor or np nd array to avoid warning
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return {"img": torch.tensor(img), "label": label}