import argparse
import os

import torch
from flax.training.common_utils import shard_prng_key, shard
from tqdm import tqdm
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib, train_state, orbax_utils
from dataset import generator
from typing import *
from functools import partial
import optax
from utils import *
from loss import l1_loss, l2_loss

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


class AutoEncoderTrainState(train_state.TrainState):
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    ema_params: Any = None


def create_state(rng, model_cls, input_shape, optimizer, train_state=AutoEncoderTrainState, print_model=True,
                 optimizer_kwargs=None, model_kwargs=None):
    platform = jax.local_devices()[0].platform

    if platform == "gpu":
        dynamic_scale = dynamic_scale_lib.DynamicScale()
        dynamic_scale = None
    else:
        dynamic_scale = None

    model = model_cls(**model_kwargs)
    if print_model:
        print(model.tabulate(rng, jnp.empty(input_shape), jnp.empty((input_shape[0],)), depth=2,
                             console_kwargs={'width': 200}))
    variables = model.init(rng, jnp.empty(input_shape))

    if optimizer == 'AdamW':
        optimizer = optax.adamw
    elif optimizer == "Lion":
        optimizer = optax.lion
    else:
        assert "some thing is wrong"

    tx = optax.chain(
        optax.clip_by_global_norm(1),
        optimizer(**optimizer_kwargs)
    )
    return train_state.create(apply_fn=model.apply, params=variables['params'], tx=tx, dynamic_scale=dynamic_scale,
                              ema_params=variables['params'])


@partial(jax.pmap, axis_name='batch')  # static_broadcasted_argnums=(3),
def train_step(state: AutoEncoderTrainState, batch, ):
    def loss_fn(params):
        reconstruct = state.apply_fn({'params': params}, batch)
        loss = l1_loss(reconstruct, batch)
        return loss.mean()

    dynamic_scale = state.dynamic_scale
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, )  # axis_name=pmap_axis
        dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
        # grad_fn = dynamic_scale.value_and_grad(cls.p_loss, argnums=1)  # axis_name=pmap_axis
        # dynamic_scale, is_fin, loss, grads = grad_fn(state.params,state,key,batch)
    else:
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
        grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"loss": loss}
    return new_state, metric


class DownSample(nn.Module):
    dim: int
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = einops.rearrange(x, 'b  (h p1) (w p2) c ->b h w (c p1 p2)', p1=2, p2=2)
        x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        return x


class Upsample(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        x = jax.image.resize(x, shape=(b, h * 2, w * 2, c), method="nearest")
        x = nn.Conv(self.dim * 4, (3, 3), padding="SAME", dtype=self.dtype)(x)
        return x


class Block(nn.Module):
    dim: int
    dtype: str = 'bfloat16'
    groups: int = 8

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dim, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=self.groups, dtype=self.dtype)(x)
        x = nn.silu(x)
        return x


class ResBlock(nn.Module):
    dim: int
    dtype: str

    @nn.compact
    def __call__(self, x):
        b, _, _, c = x.shape
        hidden = Block(self.dim, self.dtype)(x)
        hidden = Block(self.dim, self.dtype)(hidden)

        if c != self.dim:
            x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        return hidden + x


class EncoderBlock(nn.Module):
    dim: int
    num_blocks: int
    add_down: bool
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for _ in range(self.num_blocks):
            x = ResBlock(self.dim, self.dtype)(x)

        if self.add_down:
            x = DownSample(self.dim, self.dtype)(x)

        return x


class DecoderBlock(nn.Module):
    dim: int
    num_blocks: int
    add_up: bool
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):

        for _ in range(self.num_blocks):
            x = ResBlock(self.dim, self.dtype)(x)

        if self.add_up:
            x = Upsample(self.dim, self.dtype)(x)
        return x


class AutoEncoder(nn.Module):
    dims: Sequence = (64, 128, 256)
    num_blocks: int = 2
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for i, dim in enumerate(self.dims):
            x = EncoderBlock(dim, self.num_blocks, True if i != len(self.dims) else False,
                             dtype=self.dtype)(x)
        x = nn.tanh(x)
        reversed_dims = list(reversed(self.dims))

        for i, dim in enumerate(reversed_dims):
            x = DecoderBlock(dim, self.num_blocks, add_up=True if i != len(self.dims) else False,
                             dtype=self.dtype)(x)

        x = ResBlock(reversed_dims[-1], self.dtype)(x)
        x = nn.Conv(3, (3, 3), padding="SAME", dtype='float32')(x)
        x = nn.tanh(x)
        return x


def sample_save_image(state: AutoEncoderTrainState, save_path,steps,data):
    os.makedirs(save_path, exist_ok=True)
    @jax.pmap
    def infer(state,data):
        sample = state.apply_fn({'params': state.ema_params}, data)
        return sample
    sample=infer(state,data)
    all_image=jnp.concatenate([sample,data],axis=1)
    all_image = all_image / 2 + 0.5
    all_image = einops.rearrange(all_image, 'n b h w c->(n b) c h w')
    all_image = np.array(all_image)
    all_image = torch.Tensor(all_image)

    save_image(all_image, f'{save_path}/{steps}.png')



