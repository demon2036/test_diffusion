import math

import flax.linen as nn
from typing import *
import einops
import jax.random
import optax
from flax.training import train_state

from unet_block import EncoderBlock, DecoderBlock, MidBlock
import jax.numpy as jnp
import os

# from diffusers import UNet2DModel
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, ) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class Unet(nn.Module):
    out_channels: int
    layers_per_block: int
    block_out_channels: Sequence
    dtype: str
    scale: int = 1
    precision: str = "highest"

    @nn.compact
    def __call__(self, x, time, *args, **kwargs):

        dim = self.block_out_channels[0]
        t_emb = SinusoidalPosEmb(dim)(time)

        t_emb = nn.Sequential([
            nn.Dense(dim * 4, dtype=self.dtype),
            nn.gelu,
            nn.Dense(dim * 4, dtype=self.dtype)
        ])(t_emb)

        block_out_channels = self.block_out_channels

        kernal=max(self.scale**2,7)

        #x = einops.rearrange(x, 'b  (h p1) (w p2) c->b h w (c p1 p2)', p1=self.scale, p2=self.scale)
        x = nn.Conv(block_out_channels[0], (kernal, kernal),strides=self.scale, padding="SAME", dtype=self.dtype)(x)
        temp = []

        for i, channesls in enumerate(block_out_channels):
            is_final = (i == len(block_out_channels) - 1)
            x, skip_block = EncoderBlock(channesls, self.layers_per_block, dtype=self.dtype,
                                         down=True if not is_final else False)(x, t_emb)

            temp.append(skip_block)

        prev = MidBlock(block_out_channels[-1], self.layers_per_block, dtype=self.dtype)(x, t_emb)

        skip_blocks = list(reversed(temp))
        block_out_channels = list(reversed(block_out_channels))

        for i, (skip_block, channels) in enumerate(zip(skip_blocks, block_out_channels)):
            is_final = (i == len(block_out_channels) - 1)
            x = DecoderBlock(channels, self.layers_per_block, dtype=self.dtype, up=True if not is_final else False)(
                prev, skip_block, t_emb)
            prev = x

        x = nn.GroupNorm(dtype=self.dtype)(x)
        x = nn.silu(x)
        x = nn.Conv(self.out_channels * self.scale ** 2, (3, 3), padding='SAME', dtype=self.dtype)(x)
        # x = einops.rearrange(x, 'b h w (c p1 p2)->b  (h p1) (w p2) c', p1=self.scale, p2=self.scale)
        # x = nn.Conv(128, (3, 3), padding='SAME', dtype=self.dtype)(x)
        # x = nn.Conv(self.out_channels, (3, 3), padding='SAME', dtype=self.dtype)(x)

        return x


def create_state(rng, model_cls, input_shape, learning_rate, optimizer, train_state, print_model=False,
                 model_kwargs=None, *args, ):
    platform = jax.local_devices()[0].platform

    if platform == "gpu":
        dynamic_scale = None
    else:
        dynamic_scale = None

    model = model_cls(*args, **model_kwargs)
    if print_model:
        print(model.tabulate(rng, jnp.empty(input_shape), jnp.empty((input_shape[0],)), depth=2,
                             console_kwargs={'width': 200}))
    variables = model.init(rng, jnp.empty(input_shape), jnp.empty((input_shape[0],)))

    if optimizer == 'AdamW':
        optimizer = optax.adamw
    elif optimizer == "Lion":
        optimizer = optax.lion
    else:
        assert "soem thing is wrong"

    tx = optax.chain(
        optax.clip_by_global_norm(1),
        optimizer(learning_rate, weight_decay=1e-2)
    )
    return train_state.create(apply_fn=model.apply, params=variables['params'], tx=tx, dynamic_scale=dynamic_scale,
                              ema_params=variables['params'])


if __name__ == "__main__":
    key = jax.random.PRNGKey(seed=43)
    dim = 512

    model_kwargs = {
        'out_channels': 3,
        'layers_per_block': 2,
        'block_out_channels': [
            128, 256, 256, 512
        ],
        'scale': 1,
        'dtype': 'bfloat16'
    }
    s = Unet(**model_kwargs)
    variable = s.init(key, jnp.ones((1, 32, 32, 3)), jnp.ones((1,)))
    # print(variable['params'])
    # s.dtype=jnp.float32
    print(s.dtype)
    t = s.apply({'params': variable['params']}, jnp.ones((1, 16, 16, 3)), jnp.ones((1,)))
    print(t.dtype)
