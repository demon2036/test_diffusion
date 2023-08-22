import argparse
import time
from typing import Any

import optax
from flax.jax_utils import replicate
from tqdm import tqdm
import jax.random
from data.dataset import generator
from modules.gaussian.gaussian_multi import GaussianMulti
from modules.loss.loss import l1_loss, l2_loss
from modules.state_utils import create_state, apply_ema_decay, copy_params_to_ema, ema_decay_schedule, \
    create_obj_by_config, create_state_by_config
from modules.utils import EMATrainState, create_checkpoint_manager, load_ckpt, read_yaml, update_ema, \
    sample_save_image_diffusion, get_obj_from_str, sample_save_image_diffusion_multi
import flax
import os
from functools import partial
from flax.training import orbax_utils, train_state
from flax.training.common_utils import shard, shard_prng_key
from jax_smi import initialise_tracking
from modules.gaussian.gaussian import Gaussian, extract
import jax.numpy as jnp

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


@partial(jax.pmap, static_broadcasted_argnums=(3, 4), axis_name='batch')
def train_step(state, batch, train_key, cls, gaussian_conf):
    def loss_fn(params):
        loss = cls(train_key, state, params, batch, gaussian_conf)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"loss": loss}
    return new_state, metric


class DummyClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='configs/training/DiffusionMulti/test2.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    states_conf = config['states_conf']

    c = create_obj_by_config(config['Gaussian'])
    assert type(c) == GaussianMulti

    dl = generator(**dataloader_configs)  # file_path

    batch = next(dl)
    noise = jax.random.normal(key, batch.shape)
    for i in range(1000):
        batch_size = batch.shape[0]
        time = jnp.full((batch_size,), i)

        mixed_image = c.q_sample(batch, time, noise)

        loss_x0 = l2_loss(batch, mixed_image).mean()
        loss_noise = l2_loss(mixed_image, noise)
        loss_noise_snr = (loss_noise / extract((c.snr ** 0.5), time, loss_noise.shape)).mean()

        print(loss_x0, loss_noise.mean(), loss_noise_snr)

        if loss_x0 > loss_noise_snr:
            print(time)
            break


if __name__ == "__main__":
    train()
