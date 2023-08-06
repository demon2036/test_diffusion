import flax.linen as nn
from typing import *

import jax

from modules.models.unet_block import DecoderUpBlock, EncoderDownBlock
import jax.numpy as jnp

class Encoder(nn.Module):
    dims: Sequence
    num_blocks: int = 2
    dtype: Any = 'bfloat16'
    latent: int = 4
    use_attn: bool = False
    block_type: str = 'res'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dims[0], (7, 7), dtype=self.dtype, padding="same")(x)
        for i, dim in enumerate(self.dims):
            x = EncoderDownBlock(dim, self.num_blocks,
                                 True if i != len(self.dims) - 1 else False,
                                 block_type=self.block_type,
                                 dtype=self.dtype, )(x)
        x = nn.Conv(self.latent, (1, 1), dtype=self.dtype)(x)
        x = nn.tanh(x)
        return x


class Decoder(nn.Module):
    dims: Sequence
    num_blocks: int
    dtype: str
    use_attn: bool = False
    block_type: str = 'res'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dims[0], (7, 7), dtype=self.dtype, padding="same")(x)
        for i, dim in enumerate(self.dims):
            x = DecoderUpBlock(dim, self.num_blocks,
                               True if i != len(self.dims) - 1 else False,
                               block_type=self.block_type,
                               dtype=self.dtype, )(x)
        x = nn.Conv(3, (3, 3), dtype='float32', padding='SAME')(x)
        x = nn.tanh(x)
        return x


class AutoEncoder(nn.Module):
    dims: Sequence = (64, 128, 256)
    num_blocks: int = 2
    dtype: str = 'bfloat16'
    latent: int = 3
    block_type: str = 'res'

    # use_attn:bool =False

    def setup(self) -> None:
        self.encoder = Encoder(self.dims, self.num_blocks, self.dtype, self.latent, block_type=self.block_type,
                               name='Encoder_0')
        reversed_dims = list(reversed(self.dims))
        self.decoder=Decoder(reversed_dims, self.num_blocks, self.dtype, block_type=self.block_type,name='Decoder_0' )

    def encode(self,x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = self.encoder(x)
        x = self.decoder(x)
        return x










class AutoEncoderKL(nn.Module):
    dims: Sequence = (64, 128, 256)
    num_blocks: int = 2
    dtype: str = 'bfloat16'
    latent: int = 3
    block_type: str = 'res'

    # use_attn:bool =False

    def setup(self) -> None:
        self.encoder = Encoder(self.dims, self.num_blocks, self.dtype, self.latent*2, block_type=self.block_type,
                               name='Encoder_0')
        reversed_dims = list(reversed(self.dims))
        self.decoder=Decoder(reversed_dims, self.num_blocks, self.dtype, block_type=self.block_type,name='Decoder_0' )

    def encode(self,x,z_rng):
        x=self.encoder(x)

        mean, variance = jnp.split(x, 2, 3)
        self.sow('intermediate', 'mean', mean)
        self.sow('intermediate', 'variance', variance)
        z=self.reparameter(z_rng,mean,variance)
        return z
    def decode(self, x):
        return self.decoder(x)
    @nn.compact
    def __call__(self, x,z_rng, *args, **kwargs):
        z = self.encode(x,z_rng)
        x = self.decoder(z)
        return x

    def reparameter(self,rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mean + eps * std


