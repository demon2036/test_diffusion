import os
import time
import random

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

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def get_dataloader(batch_size=32, file_path='/home/john/data/s', cache=False, image_size=64, repeat=1, drop_last=True,
                   data_type='img'):
    data = MyDataSet(file_path, cache, image_size, repeat=repeat, data_type=data_type)

    dataloader = DataLoader(data, batch_size=batch_size,
                            num_workers=jax.device_count() * 2
                            , persistent_workers=True, pin_memory=True, shuffle=True,
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

        self.img_names = os.listdir(self.path)#[:10000]
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


def generator(batch_size=32, file_path='/home/john/datasets/celeba-128/celeba-128', image_size=64, cache=False,
              data_type='img',repeat=1):
    d = get_dataloader(batch_size, file_path, cache=cache, image_size=image_size, data_type=data_type, repeat=repeat)
    while True:
        for data in d:
            data = torch_to_jax(data)
            yield data


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


import jax.numpy as jnp
from jax import random, vmap, lax


def random_crop_single(rng_key, image, crop_size):
    image_height, image_width, _ = image.shape
    crop_height, crop_width = crop_size

    if image_height < crop_height or image_width < crop_width:
        raise ValueError("Crop size must be smaller than image dimensions")

    max_y = image_height - crop_height
    max_x = image_width - crop_width

    offset_y = random.randint(rng_key, (), 0, max_y + 1)
    offset_x = random.randint(rng_key, (), 0, max_x + 1)

    cropped_image = lax.dynamic_slice(image, (offset_y, offset_x, 0), (crop_height, crop_width, 3))

    return cropped_image


def random_crop_batch(rng_key, images, crop_size):
    num_images = images.shape[0]

    # Use vmap to apply random_crop_single to each image in the batch
    rng_keys = random.split(rng_key, num_images)
    cropped_images = vmap(random_crop_single, (0, 0, None))(rng_keys, images, crop_size)

    return cropped_images


# Example usage


if __name__ == '__main__':
    start = time.time()
    image_size = 256
    dl = get_dataloader(16, '/home/john/data/s', cache=False, image_size=image_size, repeat=2)
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
