import random

import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import torch
from jax import config
import math
import numpy as np
from PIL import Image

random.seed(42)


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return jnp.linspace(beta_start, beta_end, timesteps)  # , dtype = jnp.float64


# def linear_beta_schedule(timesteps):
#     """
#     linear schedule, proposed in original ddpm paper
#     """
#     scale = 1000 / timesteps
#     beta_start = scale * 0.0015
#     beta_end = scale * 0.0195
#     return jnp.linspace(beta_start, beta_end, timesteps)  # , dtype = jnp.float64


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps, dtype=jnp.float64) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps, dtype=jnp.float64) / timesteps
    v_start = jax.nn.sigmoid(jnp.array(start / tau))
    v_end = jax.nn.sigmoid(jnp.array(end / tau))
    alphas_cumprod = (- jax.nn.sigmoid(((t * (end - start) + start) / tau)) + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def generate_nosie(key, shape):
    discount = 1
    h, w, c = shape
    key, noise_key = jax.random.split(key, 2)
    noise = jax.random.normal(noise_key, shape)
    for i in range(100):
        key, noise_key = jax.random.split(key, 2)
        r = 2  # random.random() * 2 +
        print(r)
        w, h = max(1, int(w / (r))), max(1, int(h / (r)))  #
        new_shape = (w, h, c)
        new_noise = jax.random.normal(noise_key, new_shape, )
        noise += jax.image.resize(new_noise, shape, method='bilinear') * discount ** i
        print(w, h)
        if w == 1 or h == 1: break
    return noise / noise.std() * jnp.log(i)


def generate_nosie(key, shape):
    discount = 1
    h, w, c = shape
    key, noise_key = jax.random.split(key, 2)
    noise = jax.random.normal(noise_key, shape)

    return noise / noise.std()


if __name__ == "__main__":
    betas = linear_beta_schedule(1000)
    alphas = 1. - betas
    alphas_cumprod = jnp.cumprod(alphas, )
    # print(alphas_cumprod)
    img = Image.open('/home/john/datasets/celeba-128/celeba-128/183375.jpg.jpg')

    img = np.array(img)
    img = jnp.array(img)
    scale = 4
    img = jax.image.resize(img, method="bilinear", shape=(32 * scale, 32 * scale, 3))
    img = img / 255

    t = 100
    alphas = alphas_cumprod[t]
    seed = jax.random.key(42)

    # noise = jax.random.normal(seed, img.shape)
    noise = generate_nosie(seed, img.shape)
    x = jnp.sqrt(alphas) * img + jnp.sqrt(1 - alphas) * noise

    x=x/x.std()


    x = np.array(x)
    plt.subplot(121)
    plt.imshow(x)
    plt.subplot(122)
    plt.imshow(noise)

    plt.show()
