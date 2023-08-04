import os
import time
import einops
import flax.linen
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

from diffusers.schedulers import EulerDiscreteScheduler


def get_dataloader(batch_size=32, file_path='/home/john/data/s', cache=False, image_size=64, repeat=1, drop_last=True,
                   data_type='img'):
    data = MyDataSet(file_path, cache, image_size, repeat=repeat, data_type=data_type)

    dataloader = DataLoader(data, batch_size=batch_size,
                            num_workers=jax.device_count() * 2
                            , persistent_workers=False, pin_memory=False, shuffle=True,
                            drop_last=drop_last)
    return dataloader


class MyDataSet(Dataset):
    def __init__(self, path, cache=True, image_size=64, repeat=1, data_type='img'):
        self.repeat = repeat
        self.image_size = image_size
        self.path = path
        self.cache = cache
        self.data = []
        self.count = 0
        self.img_names = os.listdir(self.path)
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
            img = 2 * img - 1
            img = A.smallest_max_size(img, self.image_size, interpolation=cv2.INTER_AREA)
            img = A.center_crop(img, self.image_size, self.image_size)
            return img
        elif self.data_type == 'np':

            #with open(data_path,'r') as f:
            latent=np.load(data_path)
            try:
                latent = np.array(latent,dtype=np.float32)
                #print(latent.shape)
            except Exception as e:
                print(latent.shape,type(latent),data_path)
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


def generator(batch_size=32, file_path='/home/john/datasets/celeba-128/celeba-128', image_size=64, cache=False,
              data_type='img'):
    d = get_dataloader(batch_size, file_path, cache=cache, image_size=image_size, data_type=data_type)
    while True:
        for data in d:
            yield torch_to_jax(data)


def torch_to_jax(x):
    x = np.array(x)
    x = jnp.asarray(x)
    return x


def split_array_into_overlapping_patches(arr, patch_size, stride):
    # Get the array's shape
    batch_size, height, width, num_channels = arr.shape
    num_patches_vertical = (height - patch_size) // stride + 1
    num_patches_horizontal = (width - patch_size) // stride + 1

    # Create an array of indices for extracting patches
    y_indices = stride * jnp.arange(num_patches_vertical)
    x_indices = stride * jnp.arange(num_patches_horizontal)
    yy, xx = jnp.meshgrid(y_indices, x_indices)
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    # Calculate the indices for patches extraction
    y_indices = yy + jnp.arange(patch_size)
    x_indices = xx + jnp.arange(patch_size)

    # Extract the patches using advanced indexing
    patches = arr[:, y_indices[:, :, None], x_indices[:, None, :]]

    return patches


if __name__ == '__main__':
    start = time.time()
    dl = get_dataloader(1, '/home/john/data/s', cache=False, image_size=256, repeat=2)
    end = time.time()
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    # for _ in range(100):
    #     for data in dl:
    #         print(type(data))
    #         # print(data.shape)

    for data in dl:
        print(data.shape)
        data = data / 2 + 0.5
        data = data.numpy()
        y = jnp.asarray(data)

        data = jnp.array(split_array_into_overlapping_patches(y, 16, 16))

        data = np.asarray(data)
        data = torch.Tensor(data)
        print(data.shape)

        data = einops.rearrange(data, 'b (n) h w c->(b n) c w h', )
        torchvision.utils.save_image(data, './test2.png', nrow=256 // 16)

        data = np.asarray(y)
        data = torch.Tensor(data)
        print(data.shape)

        data = einops.rearrange(data, 'b h w c->b c h w', )
        torchvision.utils.save_image(data, './test3.png', nrow=256 // 16)
        break

    print(end - start)
