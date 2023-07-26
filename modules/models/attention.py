import flax.linen as nn


class Attention(nn.Module):
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
