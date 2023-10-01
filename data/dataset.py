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


def cutmix(rng: chex.PRNGKey,
           images: chex.Array,
           labels: chex.Array,
           alpha: float = 1.,
           beta: float = 1.,
           split: int = 1) -> Tuple[chex.Array, chex.Array]:
    """Composing two images by inserting a patch into another image."""
    batch_size, height, width, _ = images.shape
    split_batch_size = batch_size // split if split > 1 else batch_size

    # Masking bounding box.
    box_rng, lam_rng, rng = jax.random.split(rng, num=3)
    lam = jax.random.beta(lam_rng, a=alpha, b=beta, shape=())
    cut_rat = jnp.sqrt(1. - lam)
    cut_w = jnp.array(width * cut_rat, dtype=jnp.int32)
    cut_h = jnp.array(height * cut_rat, dtype=jnp.int32)
    box_coords = _random_box(box_rng, height, width, cut_h, cut_w)
    # Adjust lambda.
    lam = 1. - (box_coords[2] * box_coords[3] / (height * width))
    idx = jax.random.permutation(rng, split_batch_size)

    def _cutmix(x, y):
        images_a = x
        print(x.shape)
        images_b = x[idx, :, :, :]
        # y = lam * y + (1. - lam) * y[idx, :]
        x = _compose_two_images(images_a, images_b, box_coords)
        return x, y

    print(idx)
    if split <= 1:
        return _cutmix(images, labels)
    return None
    # Apply CutMix separately on each sub-batch. This reverses the effect of
    # `repeat` in datasets.

    images = einops.rearrange(images, '(b1 b2) ... -> b1 b2 ...', b2=split)
    labels = einops.rearrange(labels, '(b1 b2) ... -> b1 b2 ...', b2=split)
    images, labels = jax.vmap(_cutmix, in_axes=1, out_axes=1)(images, labels)
    images = einops.rearrange(images, 'b1 b2 ... -> (b1 b2) ...', b2=split)
    labels = einops.rearrange(labels, 'b1 b2 ... -> (b1 b2) ...', b2=split)
    return images, labels


def _random_box(rng: chex.PRNGKey,
                height: chex.Numeric,
                width: chex.Numeric,
                cut_h: chex.Array,
                cut_w: chex.Array) -> chex.Array:
    """Sample a random box of shape [cut_h, cut_w]."""
    height_rng, width_rng = jax.random.split(rng)
    i = jax.random.randint(
        height_rng, shape=(), minval=0, maxval=height, dtype=jnp.int32)
    j = jax.random.randint(
        width_rng, shape=(), minval=0, maxval=width, dtype=jnp.int32)
    bby1 = jnp.clip(i - cut_h // 2, 0, height)
    bbx1 = jnp.clip(j - cut_w // 2, 0, width)
    h = jnp.clip(i + cut_h // 2, 0, height) - bby1
    w = jnp.clip(j + cut_w // 2, 0, width) - bbx1
    return jnp.array([bby1, bbx1, h, w])


def _compose_two_images(images: chex.Array,
                        image_permutation: chex.Array,
                        bbox: chex.Array) -> chex.Array:
    """Inserting the second minibatch into the first at the target locations."""

    def _single_compose_two_images(image1, image2):
        height, width, _ = image1.shape
        mask = _window_mask(bbox, (height, width))
        print(mask)
        print(mask.shape)
        return image1 * (1. - mask) + image2 * mask

    return jax.vmap(_single_compose_two_images)(images, image_permutation)


def _window_mask(destination_box: chex.Array,
                 size: Tuple[int, int]) -> jnp.ndarray:
    """Mask a part of the image."""
    height_offset, width_offset, h, w = destination_box
    h_range = jnp.reshape(jnp.arange(size[0]), [size[0], 1, 1])
    w_range = jnp.reshape(jnp.arange(size[1]), [1, size[1], 1])
    return jnp.logical_and(
        jnp.logical_and(height_offset <= h_range,
                        h_range < height_offset + h),
        jnp.logical_and(width_offset <= w_range,
                        w_range < width_offset + w)).astype(jnp.float32)


if __name__ == '__main__':
    start = time.time()
    image_size = 256
    dl = generator(16, '/home/john/data/s', cache=False, image_size=image_size, repeat=2, dataset=MyDataSet)

    from tqdm import tqdm

    #cutmix = jax.jit(cutmix)

    for x in tqdm(dl):
        print(x.shape)
        x = cutmix(jax.random.PRNGKey(0), x, 1)

        """
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
        """
        break

        pass
