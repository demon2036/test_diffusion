import os
import time
import random
from typing import Tuple

import chex
import einops
import flax.linen
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.utils
import tqdm
from torch.utils.data import DataLoader, Dataset
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
    x = np.asarray(x)
    x = jnp.asarray(x)
    return x


def jax_to_torch(x):
    x = np.asarray(x)
    x = torch.Tensor(x)
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
                            num_workers= jax.device_count() * 2,prefetch_factor=2
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


def mix_up(shape, key, alpha=0.2):
    b, h, w, c = shape
    lam = jax.random.beta(key, alpha, alpha, shape=(b, h, w, 1))
    return lam


if __name__ == '__main__':
    start = time.time()
    image_size = 256
    dl = generator(16, '/home/john/data/s', cache=False, image_size=image_size, repeat=2, dataset=MyDataSet)

    from tqdm import tqdm

    # cutmix = jax.jit(cutmix)
    for _ in tqdm(range(1)):
        data1 = next(dl)
        data2 = next(dl)
        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, data1.shape[0])
        # mix_up_label = get_cut_mix_label(data1, rngs)
        mix_up_label = mix_up(data1, data2, rng)

    print(mix_up_label.shape)
    b, h, w, c = data1.shape
    factor = 2
    # mix_up_label = einops.repeat(mix_up_label, 'b h w -> b h w k', k=3)

    data2 = jax.image.resize(jax.image.resize(data1, (b, h // factor, w // factor, c), method='bicubic'), (b, h, w, c),
                             method='bicubic')

    mixed = data1 * mix_up_label + (1 - mix_up_label) * data2

    img1 = data1[0]
    img2 = data2[0]

    mixed_image = mixed[0]
    all = jnp.stack([img1, img2, mixed_image])

    data = torch.Tensor(np.array(all))

    data = einops.rearrange(data, 'b  h w c->(b ) c h w', )
    data = data / 2 + 0.5
    torchvision.utils.save_image(data, f'{1}.png', nrow=4)

    # print(mix_up_label)
