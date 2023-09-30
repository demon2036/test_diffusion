import os
import time
import random

import einops
import flax.linen
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.utils
import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, CelebA
from torchvision.transforms import *
import os
from PIL import Image
import albumentations as A
import jax.numpy as jnp
import jax

from modules.utils import get_obj_from_str

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


class MyDataSet(Dataset):
    def __init__(self, path, cache=True, image_size=64, repeat=1, data_type='img'):
        self.repeat = repeat
        self.image_size = image_size
        self.path = path
        self.cache = cache
        self.data = []
        self.count = 0

        self.img_names = os.listdir(self.path)  # [:10000]
        self.data_type = data_type

        if self.cache:
            for _ in range(self.repeat):
                for img_name in tqdm.tqdm(self.img_names):
                    self.data.append(self._preprocess(self.path + '/' + img_name))
        else:
            self.data = self.img_names

        self.real_length = len(self.data)

    def _preprocess(self, data_path):
        if self.data_type == 'img':
            img = Image.open(data_path)
            img = np.array(img) / 255.0
            img = A.smallest_max_size(img, self.image_size, interpolation=cv2.INTER_AREA)
            img = A.center_crop(img, self.image_size, self.image_size)
            img = 2 * img - 1
            return img
        elif self.data_type == 'np':
            latent = np.load(data_path)
            try:
                latent = np.array(latent, dtype=np.float32)
            except Exception as e:
                print(latent.shape, type(latent), data_path)
            return latent
        # img = A.random_crop(img, self.image_size, self.image_size, 0, 0)

    def __len__(self):
        return len(self.img_names) * self.repeat

    def __getitem__(self, idx):
        if self.cache:
            img = self.data[idx]
        else:
            img = self._preprocess(self.path + '/' + self.img_names[idx % self.real_length])

        return img


def torch_to_jax(x):
    x = np.array(x)
    x = jnp.asarray(x)
    return x


class SRDataSet(Dataset):
    def __init__(self, path, cache=True, image_size=64, repeat=1, data_type='img'):
        self.repeat = repeat
        self.image_size = image_size
        self.path = path
        self.cache = cache
        self.data = []
        self.count = 0

        self.img_names = os.listdir(self.path)  # [:10000]
        self.data_type = data_type

        if self.cache:
            for _ in range(self.repeat):
                for img_name in tqdm.tqdm(self.img_names):
                    self.data.append(self._preprocess(self.path + '/' + img_name))
        else:
            self.data = self.img_names

        self.real_length = len(self.data)

    def _preprocess(self, data_path):

        img = Image.open(data_path)
        img = np.array(img) / 255.0
        img = A.smallest_max_size(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = A.center_crop(img, self.image_size, self.image_size)
        img = 2 * img - 1

        sr_factor = np.random.uniform(low=1, high=16)
        hr_image = img
        sr_image = A.resize(hr_image, int(self.image_size // sr_factor), int(self.image_size // sr_factor))
        fake_image = A.resize(sr_image, self.image_size, self.image_size)
        return fake_image, hr_image, sr_factor

    def __len__(self):
        return len(self.img_names) * self.repeat

    def __getitem__(self, idx):
        if self.cache:
            img = self.data[idx]
        else:
            img = self._preprocess(self.path + '/' + self.img_names[idx % self.real_length])

        return img


def get_dataloader(batch_size=32, file_path='/home/john/data/s', image_size=64, cache=False, data_type='img', repeat=1,
                   drop_last=True,
                   shuffle=True,
                   dataset=MyDataSet):
    if isinstance(dataset, str):
        dataset = get_obj_from_str(dataset)

    data = dataset(file_path, cache, image_size, repeat=repeat, data_type=data_type)

    dataloader = DataLoader(data, batch_size=batch_size,
                            num_workers=jax.device_count() * 2
                            , persistent_workers=True, pin_memory=True, shuffle=shuffle,
                            drop_last=drop_last)
    return dataloader


def generator(batch_size=32, file_path='/home/john/datasets/celeba-128/celeba-128', image_size=64, cache=False,
              data_type='img', repeat=1, drop_last=True, shuffle=True, dataset=MyDataSet):
    d = get_dataloader(batch_size, file_path, cache=cache, image_size=image_size, data_type=data_type, repeat=repeat,
                       drop_last=True, shuffle=True, dataset=dataset)
    while True:
        for data in d:

            if isinstance(data, list):
                data = [torch_to_jax(sub_data) for sub_data in data]
            else:
                data = torch_to_jax(data)
            yield data


import jax
import jax.numpy as jnp


def random_boundingbox(size, lam):
    width, height = size, size

    r = np.sqrt(1. - lam)
    w = np.int32(width * r)
    h = np.int32(height * r)
    x = np.random.randint(width)
    y = np.random.randint(height)

    x1 = np.clip(x - w // 2, 0, width)
    y1 = np.clip(y - h // 2, 0, height)
    x2 = np.clip(x + w // 2, 0, width)
    y2 = np.clip(y + h // 2, 0, height)

    return x1, y1, x2, y2


def CutMix(imsize):
    lam = np.random.beta(1, 1)
    x1, y1, x2, y2 = random_boundingbox(imsize, lam)
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (imsize * imsize))
    map = torch.ones((imsize, imsize))
    map[x1:x2, y1:y2] = 0
    if torch.rand(1) > 0.5:
        map = 1 - map
        lam = 1 - lam
    # lam is equivalent to map.mean()
    return map  # , lam


if __name__ == '__main__':
    """
    start = time.time()
    image_size = 256
    dl = get_dataloader(16, '/home/john/data/s', cache=False, image_size=image_size, repeat=2, dataset=MyDataSet)

    from tqdm import tqdm

    for x in tqdm(dl):
        cut_mix_map = jnp.asarray(CutMix(image_size))
        cut_mix_map=einops.repeat(cut_mix_map,'h w -> h w k',k=3)

        x = np.asarray(x)
        x = jnp.asarray(x)

        img1 = x[0]
        img2 = x[1]

        mixed_image = img1 * cut_mix_map + (1 - cut_mix_map) * img2
        all = jnp.stack([img1, img2, mixed_image])

        data = torch.Tensor(np.array(all))

        data = einops.rearrange(data, 'b  h w c->(b ) c h w', )
        torchvision.utils.save_image(data, f'{1}.png', nrow=4)

        break

        pass

    
    
    end = time.time()
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    os.environ['XLA_FLAGS'] = 'TF_USE_NVLINK_FOR_PARALLEL_COMPILATION=0'
    # os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

    rng_key = random.PRNGKey(0)
    count = 0
    for data in dl:
        data = data / 2 + 0.5
        data = data.numpy()
        y = jnp.asarray(data)
        print(data.shape)

        crop_size = (128, 128)

        # Create a random PRNG key
        rng_key, crop_rng = jax.random.split(rng_key, 2)

        crop = jax.random.choice(crop_rng, jnp.array([64, 256]), p=jnp.array([0.1, 0.9]))
        crop_size = (crop, crop)

        # Generate random crops for the batch of images
        cropped_images = random_crop_batch(rng_key, data, crop_size)

        data = cropped_images

        data = np.asarray(data)
        data = torch.Tensor(data)
        print(data.shape)

        data = einops.rearrange(data, 'b  h w c->(b ) c h w', )
        torchvision.utils.save_image(data, f'./test/{count}.png', nrow=4)
        count += 1

    print(end - start)
    """
