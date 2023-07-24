import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import *


class EMATrainState(train_state.TrainState):
    batch_stats:Any=None
    ema_params: Any = None


def create_discriminator_state(rng, input_shape, optimizer, train_state=EMATrainState, print_model=True,
                 optimizer_kwargs=None,):
    model = NLayerDiscriminator()
    if print_model:
        print(model.tabulate(rng, jnp.empty(input_shape),False, depth=2,
                             console_kwargs={'width': 200}))
    variables = model.init(rng, jnp.empty(input_shape))
    if optimizer == 'AdamW':
        optimizer = optax.adamw
    elif optimizer == "Lion":
        optimizer = optax.lion
    else:
        assert "some thing is wrong"

    tx = optax.chain(
        optax.clip_by_global_norm(1),
        optimizer(**optimizer_kwargs)
    )


    return train_state.create(apply_fn=model.apply, params=variables['params'], tx=tx, batch_stats=variables['batch_stats'],
                              ema_params=variables['params'] )


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    input_nc: int = 3
    ndf: int = 64
    n_layers: int = 3
    use_actnorm: bool = False

    @nn.compact
    def __call__(self, x, train: bool=True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        norm_layer = nn.BatchNorm
        x = nn.Conv(self.ndf, (4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        for n in range(1, self.n_layers):
            nf_mult = min(2 ** n, 8)
            x = nn.Conv(self.ndf * nf_mult, (4, 4), (2, 2), "SAME", use_bias=False)(x)
            x = norm_layer(use_running_average=not train)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(1, (4, 4), padding="SAME")(x)
        return x
