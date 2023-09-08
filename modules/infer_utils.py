import os
import einops
import numpy as np
import torch
from torchvision.utils import save_image
import jax
import jax.numpy as jnp
from modules.gaussian.gaussian import Gaussian
from modules.gaussian.gaussianDecoder import GaussianDecoder
from modules.gaussian.gaussianSR import GaussianSR
from modules.gaussian.gaussian_multi import GaussianMulti
from modules.gaussian.gaussian_test import GaussianTest
from modules.models.autoencoder import AutoEncoder
from modules.models.diffEncoder import DiffEncoder
from modules.state_utils import EMATrainState


def sample_save_image_autoencoder(state, save_path, steps, data, z_rng):
    os.makedirs(save_path, exist_ok=True)

    @jax.pmap
    def infer(state, params, data):
        sample = state.apply_fn({'params': params}, data, z_rng=z_rng)
        return sample

    if steps < 50000:
        sample = infer(state, state.params, data)
    else:
        sample = infer(state, state.ema_params, data)

    all_image = jnp.concatenate([sample, data], axis=1)
    all_image = all_image / 2 + 0.5
    all_image = einops.rearrange(all_image, 'n b h w c->(n b) c h w')
    all_image = np.array(all_image)
    all_image = torch.Tensor(all_image)
    save_image(all_image, f'{save_path}/{steps}.png')


def sample_save_image_diffusion_encoder(key, c: GaussianDecoder, steps, state: EMATrainState, save_path, batch):
    os.makedirs(save_path, exist_ok=True)
    print(batch.shape)
    c.eval()
    sample = c.sample(key, state, batch)
    c.train()
    sample = jnp.concatenate([sample, batch], axis=0)
    sample = sample / 2 + 0.5

    print(sample.shape)
    sample = einops.rearrange(sample, '(n b) h w c->(b n) c h w', n=2)
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'{save_path}/{steps}.png')


def sample_save_image_diffusion(key, c: Gaussian, steps, state: EMATrainState, save_path, batch_size):
    os.makedirs(save_path, exist_ok=True)
    c.eval()
    sample = c.sample(key, state, batch_size=batch_size)
    c.train()
    return sample


def jax_img_save(img, save_path, steps):
    img = img / 2 + 0.5
    img = einops.rearrange(img, 'b h w c->b c h w')
    img = np.array(img)
    img = torch.Tensor(img)
    save_image(img, f'{save_path}/{steps}.png')


def sample_save_image_diffusion_multi(key, c: GaussianMulti, steps, state: EMATrainState, save_path, gaussian_configs):
    os.makedirs(save_path, exist_ok=True)
    c.eval()
    sample = c.sample(key, state, batch_size=64, gaussian_configs=gaussian_configs)
    c.train()
    sample = sample / 2 + 0.5
    sample = einops.rearrange(sample, 'b h w c->b c h w')
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'{save_path}/{steps}.png')


@jax.pmap
def encode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=AutoEncoder.encode)


@jax.pmap
def decode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=AutoEncoder.decode)


def sample_save_image_latent_diffusion(key, c, steps,
                                       state: EMATrainState, save_path, ae_state: EMATrainState,
                                       first_stage_gaussian: GaussianTest):
    os.makedirs(save_path, exist_ok=True)
    c.eval()
    sample_latent = c.sample(key, state, batch_size=16)
    c.train()
    print(sample_latent.shape)
    first_stage_gaussian.eval()
    sample = first_stage_gaussian.sample(key, ae_state, sample_latent)
    print(sample.shape)
    sample = sample / 2 + 0.5
    sample = einops.rearrange(sample, 'b h w c->( b) c h w')
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'{save_path}/{steps}.png')


@jax.pmap
def diff_encode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=DiffEncoder.encode)


def sample_save_image_sr_eval(key, diffusion: GaussianSR, steps, state: EMATrainState, batch, save_path):
    os.makedirs(save_path, exist_ok=True)
    b, h, w, c = batch.shape
    lr_image = jax.image.resize(batch, (b, h // diffusion.sr_factor, w // diffusion.sr_factor, c), method='bilinear')
    os.makedirs(save_path, exist_ok=True)
    diffusion.set_state(state)
    sample = diffusion.sample(key, state, lr_image)
    sample.append(batch)
    all_image = jnp.concatenate(sample, axis=0)
    all_image = all_image / 2 + 0.5
    diffusion.state = None
    all_image = einops.rearrange(all_image, '(n b) h w c->b c (n h) w', n=len(sample))
    all_image = np.array(all_image)
    all_image = torch.Tensor(all_image)
    save_image(all_image, f'{save_path}/{steps}.png')


def sample_save_image_sr(key, diffusion: GaussianSR, state: EMATrainState, lr_image):
    sample = diffusion.sample(key, state, lr_image,return_img_only=True)
    return sample
