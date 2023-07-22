from typing import Optional

import numpy as np
from flax.training import dynamic_scale as dynamic_scale_lib
from dataset import get_dataloader
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import einops
from torchvision.utils import save_image
from functools import partial
from orbax import checkpoint
from orbax.checkpoint import CheckpointManagerOptions, CheckpointManager, PyTreeCheckpointer
import orbax

import yaml
import json


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


def ds(batch_size=8, size=256):
    dataloader = get_dataloader(batch_size=batch_size, size=size)
    while True:
        for data in dataloader:
            yield data


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


class test_layer(nn.Module):
    @nn.compact
    def __call__(self, x, rng, *args, **kwargs):
        print(rng)
        m = jax.random.normal(rng, (1, 4))

        return x


if __name__ == '__main__':
    pass
