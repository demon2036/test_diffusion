import time
from typing import Any
import os
import einops
import flax.linen as nn
import jax.numpy as jnp
import jax.random
from einops import rearrange

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


"""
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
"""


def l2norm(t, axis=1, eps=1e-12):
    """Performs L2 normalization of inputs over specified axis.

    Args:
      t: jnp.ndarray of any shape
      axis: the dimension to reduce, default -1
      eps: small value to avoid division by zero. Default 1e-12
    Returns:
      normalized array of same shape as t


    """
    denom = jnp.clip(jnp.linalg.norm(t, ord=2, axis=axis, keepdims=True), eps)
    out = t / denom
    return (out)


class Attention(nn.Module):
    dim: int
    scale: int = 10
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(epsilon=1e-5, use_bias=False,dtype=self.dtype)(x)
        B, H, W, C = x.shape
        dim = self.dim  # self.dim_head * self.heads
        dim_head = 64
        heads = dim // dim_head

        qkv = nn.Conv(features=dim * 3, kernel_size=(1, 1),
                      use_bias=False, dtype=self.dtype, name='to_qkv.conv_0')(x)  # [B, H, W, dim *3]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [B, H, W, dim]
        q, k, v = map(lambda t: rearrange(
            t, 'b x y (h d) -> b (x y) h d', h=heads), (q, k, v))

        assert q.shape == k.shape == v.shape == (
            B, H * W, heads, dim_head)

        q, k = map(l2norm, (q, k))

        sim = jnp.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        attn = nn.softmax(sim, axis=-1)
        assert attn.shape == (B, heads, H * W, H * W)

        out = jnp.einsum('b h i j , b j h d  -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=H)
        assert out.shape == (B, H, W, dim)

        out = nn.Conv(features=C, kernel_size=(1, 1), dtype=self.dtype, name='to_out.conv_0')(out)
        return (out)


if __name__ == "__main__":
    pass
    """
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
    """