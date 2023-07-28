import os
import importlib
import einops
import numpy as np
import torch
from torchvision.utils import save_image
from flax.training import train_state
from typing import Any
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from orbax import checkpoint
import orbax
import yaml
import json

from modules.gaussian.gaussian import Gaussian
from modules.gaussian.gaussianDecoder import GaussianDecoder
from modules.gaussian.gaussianSR import GaussianSR


class EMATrainState(train_state.TrainState):
    batch_stats:Any=None
    ema_params: Any = None


def read_yaml(config_path):
    with open(config_path, 'r') as f:
        res = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(res, indent=5))
        return res


@partial(jax.pmap, static_broadcasted_argnums=(1,))
def update_ema(state, ema_decay=0.999):
    new_ema_params = jax.tree_map(lambda ema, normal: ema * ema_decay + (1 - ema_decay) * normal, state.ema_params,
                                  state.params)
    state = state.replace(ema_params=new_ema_params)
    return state





def create_checkpoint_manager(save_path, max_to_keep=10, ):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        save_path, orbax_checkpointer, options)
    return checkpoint_manager


def load_ckpt(checkpoint_manager: orbax.checkpoint.CheckpointManager, model_ckpt):
    step = checkpoint_manager.latest_step()
    print(step)

    raw_restored = checkpoint_manager.restore(step, items=model_ckpt)
    return raw_restored


def hinge_d_loss(logits_real, logits_fake):
    loss_real = jnp.mean(nn.relu(1. - logits_real))
    loss_fake = jnp.mean(nn.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
            jnp.mean(nn.softplus(-logits_real)) +
            jnp.mean(nn.softplus(logits_fake)))
    return d_loss


def sample_save_image_autoencoder(state, save_path, steps, data):
    os.makedirs(save_path, exist_ok=True)
    @jax.pmap
    def infer(state, params, data):
        sample = state.apply_fn({'params': params}, data)
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




def sample_save_image_diffusion_encoder(key, c:GaussianDecoder, steps, state: EMATrainState,save_path,batch):
    os.makedirs(save_path, exist_ok=True)
    c.set_state(state)
    sample = c.sample(key, state,batch, batch_size=64)
    c.state = None
    sample = jnp.concatenate([sample, batch], axis=0)
    sample = sample / 2 + 0.5
    sample = einops.rearrange(sample, '(b n) h w c->(n b) c h w',n=2)
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'{save_path}/{steps}.png')





def sample_save_image_diffusion(key, c:Gaussian, steps, state: EMATrainState,save_path):
    os.makedirs(save_path, exist_ok=True)
    c.set_state(state)
    sample = c.sample(key, state, batch_size=64)
    sample = sample / 2 + 0.5
    c.state = None
    sample = einops.rearrange(sample, 'b h w c->b c h w')
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'{save_path}/{steps}.png')

def sample_save_image_sr(key, diffusion:GaussianSR, steps, state: EMATrainState, batch, save_path):
    os.makedirs(save_path,exist_ok=True)
    b, h, w, c = batch.shape
    lr_image = jax.image.resize(batch, (b, h // diffusion.sr_factor, w // diffusion.sr_factor, c), method='bilinear')
    os.makedirs(save_path, exist_ok=True)
    diffusion.set_state(state)
    sample = diffusion.sample(key, state, lr_image)
    all_image = jnp.concatenate([sample, batch], axis=0)
    sample = all_image / 2 + 0.5
    diffusion.state = None
    sample = einops.rearrange(sample, '(b n) h w c->(n b) c h w',n=2)
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'{save_path}/{steps}.png')



def get_obj_from_str(string:str):
    module,cls=string.rsplit('.',1)
    return getattr(importlib.import_module(module),cls)
