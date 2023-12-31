import jax.numpy as jnp
import flax.linen as nn
from modules.models.resnet import ResBlock, DownSample, UpSample, EfficientBlock
from modules.models.repnet import RepBlock
from modules.models.attention import Attention


class EncoderDownBlock(nn.Module):
    dim: int
    num_blocks: int
    add_down: bool
    dtype: str
    use_attn: bool = False
    block_type: str = 'res'

    @nn.compact
    def __call__(self, x, *args, **kwargs):

        if self.block_type == 'res':
            block = ResBlock
        elif self.block_type == 'rep':
            block = RepBlock
        elif self.block_type == 'efficient':
            block = EfficientBlock

        for _ in range(self.num_blocks):
            x = block(self.dim, dtype=self.dtype)(x)

        # if self.use_attn:
        #     x=Attention(self.dim,self.dtype)(x)+x

        if self.add_down:
            x = DownSample(self.dim, self.dtype)(x)
        return x


class DecoderUpBlock(nn.Module):
    dim: int
    num_blocks: int
    add_up: bool
    dtype: str
    use_attn: bool = False
    block_type: str = 'res'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        if self.block_type == 'res':
            block = ResBlock
        elif self.block_type == 'rep':
            block = RepBlock
        elif self.block_type == 'efficient':
            block = EfficientBlock


        for _ in range(self.num_blocks):
            x = block(self.dim, self.dtype)(x)

        # if self.use_attn:
        #     x=Attention(self.dim,self.dtype)(x)+x

        if self.add_up:
            x = UpSample(self.dim, self.dtype)(x)
        return x
