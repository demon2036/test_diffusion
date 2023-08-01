import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from modules.models.attention import MyAttention


class Transformer(nn.Module):
    dim: int
    dtype: Any = 'bfloat16'

    @nn.compact
    def __call__(self, x,time_emb=None):




        y = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = MyAttention(self.dim,dtype=self.dtype)(x) + y
        y = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Conv(self.dim,(1,1),dtype=self.dtype)(x) + y
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
