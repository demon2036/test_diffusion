import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from modules.models.attention import MyAttention


class MLP(nn.Module):
    dim: int
    dtype: Any = 'bflaot16'
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.dim * self.mlp_ratio, (1, 1), dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Conv(self.dim, dtype=self.dtype)(x)
        return x


class Transformer(nn.Module):
    dim: int
    dtype: Any = 'bfloat16'

    @nn.compact
    def __call__(self, x, time_emb=None):
        if time_emb is not None:
            time_emb = nn.Dense(self.dim * 6, dtype=self.dtype)(time_emb)
            time_emb = einops.rearrange(time_emb, 'b c->b 1 1 c')
            gate_msa, gate_ffn, scale_msa, scale_ffn, shift_msa, shift_ffn = jnp.split(time_emb, 3, 6)

        y = x
        x = nn.LayerNorm(dtype=self.dtype)(x)

        if time_emb is not None:
            x = x * (scale_msa + 1) + shift_msa

        attn = MyAttention(self.dim, dtype=self.dtype)(x)

        if time_emb is not None:
            attn = attn * gate_msa

        x = attn + y

        y = x

        x = nn.LayerNorm(dtype=self.dtype)(x)

        if time_emb is not None:
            x = x * (scale_ffn + 1) + shift_ffn

        ffn = MLP(self.dim, dtype=self.dtype)(x)

        if time_emb is not None:
            ffn = attn * gate_ffn

        x = ffn + y

        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    times = 1

    shape = (1, 16, 16, 100)
    x = jnp.ones(shape)
    a1 = Transformer(100)
    variable = a1.init(key, jnp.ones(shape))
    out = a1.apply({'params': variable['params']}, x)
    print(out.shape)
