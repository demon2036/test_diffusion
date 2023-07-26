import argparse
import importlib
import importlib.util
from tqdm import tqdm
import jax.random
from data.dataset import generator
from modules.models.unet import MultiUnet, Unet
from modules.state_utils import create_state
from modules.utils import EMATrainState, create_checkpoint_manager, load_ckpt, read_yaml, update_ema, \
    sample_save_image_diffusion, get_obj_from_str
import flax
import os
import time
from functools import partial
from flax.training import orbax_utils
import optax
from flax.training.common_utils import shard, shard_prng_key
from jax_smi import initialise_tracking
from modules.gaussian.gaussian import Gaussian
import jax.numpy as jnp

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'


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
    parser.add_argument('-cp', '--config_path', default='configs/Diffusion/test_diff.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, unet_config = config['Unet']['target'], config['Unet']['params']
    model_cls = get_obj_from_str(model_cls_str)
    gaussian_config = config['Gaussian']

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs, optimizer_str, optimizer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)
    input_shapes = (input_shape, input_shape[0])

    c = Gaussian(**gaussian_config, image_size=dataloader_configs['image_size'])

    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes, optimizer_name=optimizer_str,
                         optimizer_kwargs=optimizer_configs,
                         train_state=EMATrainState, model_kwargs=unet_config)

    """
    model_ckpt = {'model': state, 'steps': 0}
    model_save_path = trainer_configs['model_path']

    checkpoint_manager = create_checkpoint_manager(model_save_path, max_to_keep=10)
    if len(os.listdir(model_save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    dl = generator(**dataloader_configs)  # file_path
    finished_steps = model_ckpt['steps']

    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, 1000000):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            batch = shard(batch)
            state, metrics = train_step(state, batch, train_step_key, c)
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            pbar.set_postfix(metrics)
            pbar.update(1)

            if steps > 100 and steps % 10 == 0:
                state = update_ema(state, 0.995)

            if steps % trainer_configs['sample_steps'] == 0:
                try:
                    sample_save_image_diffusion(key, c, steps, state,save_path)
                except Exception as e:
                    print(e)

                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}  # 'steps': steps
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)
                # del unreplicate_state, sample, model_ckpt

    end = time.time()
    """
