import time
from typing import Any
import os
import einops
import flax.linen as nn
import jax.numpy as jnp
import jax.random

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


class MyAttention(nn.Module):
    dim: int
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dim, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.softmax(x, axis=(1, 2))
        x = nn.Conv(self.dim * 4, (1, 1), padding="SAME", dtype=self.dtype)(x)
        x = nn.softmax(x, axis=(-1))
        x = nn.Conv(self.dim, (3, 3), padding="SAME", dtype=self.dtype)(x)
        return x


class Attention(nn.Module):
    dim: int
    head: int = 4
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        qkv = nn.Conv(self.dim * 3, (1, 1), dtype=self.dtype, use_bias=False)(x)
        qkv = einops.rearrange(qkv, 'b x y (h d)->b (x y) h d', h=self.head)
        q, k, v = jnp.split(qkv, 3, -1)

        attn = jnp.einsum('b i h d, b j h d -> b h i j', q, k) * b ** (-0.5)

        attn = nn.softmax(attn, axis=-1)
        out = jnp.einsum('b h i j , b j h d  -> b h i d', attn, v)

        # out=attn*v
        out = einops.rearrange(out, 'b (x y) h d->b x y (h d)', x=h)
        out = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(out)
        return out


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    times = 1

    shape = (1, 1, 1, 4)
    x = jnp.ones(shape)
    a1 = Attention(4)
    a2 = Attention2(4, 1)
    variable = a1.init(key, jnp.ones(shape))

    start = time.time()
    out = a1.apply({'params': variable['params']}, x)
    print(out, out.shape)
    end = time.time()

    variable = a2.init(key, jnp.ones(shape))
    start = time.time()
    out = a2.apply({'params': variable['params']}, x)
    print(out, out.shape)
    end = time.time()
