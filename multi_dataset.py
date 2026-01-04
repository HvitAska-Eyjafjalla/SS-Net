import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image

import tqdm
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2


class MultiDataset(Dataset):
    def __init__(self, universal_config, Mylogger=None, train=False, validation=False, test=False):
        super(MultiDataset, self).__init__()
        self.universal_config = universal_config
        if train:
            sub_directory_name = 'Train_Folder'
        elif validation:
            sub_directory_name = 'Val_Folder'
        elif test:
            sub_directory_name = 'Test_Folder'
        else:
            assert 'error'


        sub_directory = self.universal_config.dataset_directory / sub_directory_name
        image_directory = sub_directory / 'img'
        mask_directory = sub_directory / 'labelcol'

        images_list = sorted([f.name for f in list(image_directory.iterdir()) if
                              f.is_file() and f.suffix.lower().lstrip('.') in ['jpg', 'jpeg', 'png', 'bmp', 'tif']])
        masks_list = sorted([f.name for f in list(mask_directory.iterdir()) if
                             f.is_file() and f.suffix.lower().lstrip('.') in ['jpg', 'jpeg', 'png', 'bmp', 'tif']])

        self.mean, self.std = self.get_dataset_mean_std(images_list, image_directory, sub_directory_name)

        log_info = f'Created dataset from {sub_directory_name}, length:{len(images_list)}, mean:{self.mean}, std:{self.std}'
        print(log_info)
        if Mylogger is not None:
            Mylogger.logger.info(log_info)

        self.data = []
        for i in range(len(images_list)):
            images_path = image_directory / images_list[i]
            mask_path = mask_directory / masks_list[i]
            self.data.append([images_path, mask_path])

        if train:
            self.transformer = transforms.Compose([
                myResize(self.universal_config.input_size_h, self.universal_config.input_size_w),
                myRandomFlip(p=0.5),
                myRandomRotation(p=0.5),
                myNormalize(mean=self.mean, std=self.std),
                myToTensor()
            ])
        else:
            self.transformer = transforms.Compose([
                myResize(self.universal_config.input_size_h, self.universal_config.input_size_w),
                myNormalize(mean=self.mean, std=self.std),
                myToTensor()
            ])

    def __getitem__(self, index):
        img_path, msk_path = self.data[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2)
        split_value = 10
        msk[msk < split_value] = 0
        msk[msk >= split_value] = 1
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)

    def get_dataset_mean_std(self, images_list, image_directory, sub_directory_name):
        image_channels = 3
        cumulative_mean = np.zeros(image_channels)
        cumulative_std = np.zeros(image_channels)

        for image_name in tqdm.tqdm(images_list, total=len(images_list),
                                    desc=f'Calculating mean and std in {sub_directory_name}'):
            image_path = image_directory / image_name
            image = np.array(Image.open(image_path))

            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()

        mean = cumulative_mean / len(images_list)
        std = cumulative_std / len(images_list)
        return mean, std


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        if image.ndimension() == 2:
            image = image.unsqueeze(0)
        elif image.ndimension() == 3:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected number of dimensions for image: {image.ndimension()}")

        if mask.ndimension() == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndimension() == 3:
            mask = mask.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected number of dimensions for mask: {mask.ndimension()}")
        return image, mask

                
class myResize:
    def __init__(self, size_h=512, size_w=512):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        image_resized = cv2.resize(image, (self.size_w, self.size_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(mask, (self.size_w, self.size_h), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return self.random_rot_flip(image, mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.p = p

    def random_rotate(self, image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return self.random_rotate(image, mask)
        else:
            return image, mask

class myNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - np.mean(img)) / np.std(img)
        img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized)))
        return img_normalized, msk


