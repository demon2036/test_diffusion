import argparse

import optax
from flax.jax_utils import replicate
from tqdm import tqdm
import jax.random
from data.dataset import generator
from modules.gaussian.gaussianDecoder import GaussianDecoder
from modules.state_utils import create_state, apply_ema_decay, copy_params_to_ema, ema_decay_schedule, \
    create_obj_by_config, create_state_by_config
from modules.utils import  create_checkpoint_manager, load_ckpt, read_yaml, update_ema, get_obj_from_str, default
import flax
import os
from functools import partial
from flax.training import orbax_utils
from flax.training.common_utils import shard, shard_prng_key
from jax_smi import initialise_tracking
from modules.gaussian.gaussian import Gaussian
import jax.numpy as jnp

from trainers.diffencoder_trainer import DiffEncoderTrainer

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


@partial(jax.pmap, static_broadcasted_argnums=(3), axis_name='batch')
def train_step(state, batch, train_key, cls):
    def loss_fn(params):
        loss = cls(train_key, state, params, batch)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"loss": loss}
    return new_state, metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/training/DiffusionEncoder/test.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_gaussian = create_obj_by_config(config['Gaussian'])
    train_state = create_state_by_config(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                         state_configs=config['State'])
    trainer = DiffEncoderTrainer(train_state, train_gaussian, **config['train'])
    trainer.load()
    # trainer.sample()
    trainer.train()
