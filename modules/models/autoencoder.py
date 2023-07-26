import flax.linen as nn
from typing import *
from modules.models.unet_block import DecoderUpBlock, EncoderDownBlock



class Encoder(nn.Module):
    dims: Sequence
    num_blocks: int
    dtype: str
    latent: int
    use_attn: bool = False


    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dims[0], (7, 7), dtype=self.dtype, padding="same")(x)
        for i, dim in enumerate(self.dims):
            x = EncoderDownBlock(dim, self.num_blocks, True if i != len(self.dims) else False,dtype=self.dtype,)(x)
        x = nn.Conv(self.latent, (1, 1), dtype=self.dtype)(x)
        x = nn.tanh(x)
        return x


class Decoder(nn.Module):
    dims: Sequence
    num_blocks: int
    dtype: str
    use_attn: bool = False

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dims[0], (7, 7), dtype=self.dtype, padding="same")(x)
        for i, dim in enumerate(self.dims):
            x = DecoderUpBlock(dim, self.num_blocks, True if i != len(self.dims) else False,dtype=self.dtype,)(x)
        x = nn.Conv(3, (3, 3), dtype='float32', padding='SAME')(x)
        x = nn.tanh(x)
        return x


class AutoEncoder(nn.Module):
    dims: Sequence = (64, 128, 256)
    num_blocks: int = 2
    dtype: str = 'bfloat16'
    latent: int = 3
    # use_attn:bool =False

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = Encoder(self.dims, self.num_blocks, self.dtype, self.latent)(x)
        reversed_dims = list(reversed(self.dims))
        x = Decoder(reversed_dims, self.num_blocks, self.dtype,)(x)
        return x


