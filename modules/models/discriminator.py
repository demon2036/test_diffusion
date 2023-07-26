from modules.utils import EMATrainState
import jax.numpy as jnp
import flax.linen as nn
import optax


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    input_nc: int = 3
    ndf: int = 64
    n_layers: int = 3
    use_actnorm: bool = False
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, train: bool = True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        norm_layer = nn.BatchNorm
        x = nn.Conv(self.ndf, (4, 4), strides=(2, 2), padding="SAME",dtype=self.dtype)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        for n in range(1, self.n_layers):
            nf_mult = min(2 ** n, 8)
            x = nn.Conv(self.ndf * nf_mult, (4, 4), (2, 2), "SAME", use_bias=True,dtype=self.dtype)(x)
            x = norm_layer(use_running_average=not train,dtype=self.dtype)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(1, (4, 4), padding="SAME")(x)
        return x
