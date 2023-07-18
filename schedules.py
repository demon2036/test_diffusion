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



def g_noise(noise_key,shape):
   # return jax.random.generalized_normal(noise_key,1,shape)
    return jax.random.normal(noise_key, shape)

def generate_nosie(key, shape):
    return g_noise(key,shape)
    discount = 0.9
    h, w, c = shape
    key, noise_key = jax.random.split(key, 2)

    noise = g_noise(noise_key,shape)
    for i in range(100):
        key, noise_key = jax.random.split(key, 2)
        r = 2  # random.random() * 2 +
        w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))  #
        new_shape = (w, h, c)
        new_noise = g_noise(noise_key,new_shape)
        noise += jax.image.resize(new_noise, shape, method='bilinear') * discount ** i

        if w == 1 or h == 1: break
    return noise / noise.std()


if __name__ == "__main__":
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    betas = cosine_beta_schedule(1000)
    alphas = 1. - betas
    alphas_cumprod = jnp.cumprod(alphas, )
    # print(alphas_cumprod)
    img = Image.open('/home/john/data/celeba-128/celeba-128/183375.jpg.jpg')

    img = np.array(img)
    img = jnp.array(img)
    scale = 8
    img = jax.image.resize(img, method="bilinear", shape=(64 * scale, 64 * scale, 3))
    img = img / 255
    img = img * 2 - 1

    t = 100
    alphas = alphas_cumprod[t]


    snr=alphas/(1-alphas)

    alphas=1- 1/(1+(1/scale)**2*snr)


    seed = jax.random.key(42)

    # noise = jax.random.normal(seed, img.shape)
    noise = g_noise(seed, img.shape)
    x = jnp.sqrt(alphas) * img + jnp.sqrt(1-alphas) * noise

    # x=x/x.std()
    print(alphas,snr)
    x = np.array(x)
   # x = x/x.std()

    print(x.max())

    x = x /2 + 0.5
    plt.subplot(121)
    plt.imshow(x)
    plt.subplot(122)
    plt.imshow(noise)

    plt.show()
