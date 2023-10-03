import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence,Any
from modules.models.resnet import DownSample, UpSample, ResBlock


class NLayerDiscriminator(nn.Module):
    input_nc: int = 3
    ndf: int = 64
    n_layers: int = 3
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, train: bool = True,*args,**kwargs):
        norm_layer = nn.BatchNorm
        x = nn.Conv(self.ndf, (4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        for n in range(1, self.n_layers):
            nf_mult = min(2 ** n, 8)
            x = nn.Conv(self.ndf * nf_mult, (4, 4), (2, 2), "SAME", use_bias=True, dtype=self.dtype)(x)
            x = norm_layer(use_running_average=not train, dtype=self.dtype)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(1, (4, 4), padding="SAME")(x)
        return x


class UnetDiscriminator(nn.Module):
    dim: int = 64
    layers_per_block: int = 1
    out_channels: int = 1
    resnet_block_groups: int = 8,
    channels: int = 3,
    dim_mults: Sequence = (1, 2, 4, 8)
    dtype: Any = jnp.bfloat16
    self_condition: bool = False

    @nn.compact
    def __call__(self, x, *args, **kwargs):

        x = nn.Conv(self.dim, (7, 7), padding="SAME", dtype=self.dtype)(x)
        r = x

        h = []

        for i, dim_mul in enumerate(self.dim_mults):
            dim = self.dim * dim_mul
            for _ in range(self.layers_per_block):
                x = ResBlock(dim, dtype=self.dtype)(x)
                h.append(x)

            if i != len(self.dim_mults) - 1:
                x = DownSample(dim, dtype=self.dtype)(x)
            else:
                x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        x = ResBlock(dim, dtype=self.dtype)(x)
        # x = self.mid_attn(x) + x
        x = ResBlock(dim, dtype=self.dtype)(x)

        reversed_dim_mults = list(reversed(self.dim_mults))

        for i, dim_mul in enumerate(reversed_dim_mults):
            dim = self.dim * dim_mul

            for _ in range(self.layers_per_block):
                x = jnp.concatenate([x, h.pop()], axis=3)
                x = ResBlock(dim, dtype=self.dtype)(x)

            if i != len(self.dim_mults) - 1:
                x = UpSample(dim, dtype=self.dtype)(x)
            else:
                x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        x = jnp.concatenate([x, r], axis=3)
        x = ResBlock(dim, dtype=self.dtype)(x)
        x = nn.Conv(self.out_channels, (1, 1), dtype="float32")(x)
        return x
