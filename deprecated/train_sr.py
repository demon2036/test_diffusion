import argparse
import jax.numpy as jnp6
import einops
import numpy as np
from tqdm import tqdm
import jax.random
import flax
from data.dataset import generator
from modules.gaussian.gaussianSR import GaussianSR
from modules.state_utils import create_state
from modules.utils import EMATrainState, load_ckpt, create_checkpoint_manager, sample_save_image_sr, update_ema, \
    read_yaml, get_obj_from_str
import os
from functools import partial
from flax.training import orbax_utils
from flax.training.common_utils import shard, shard_prng_key
from jax_smi import initialise_tracking

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


@partial(jax.pmap, static_broadcasted_argnums=(3), axis_name='batch')
def train_step(state: EMATrainState, batch, train_key, cls):
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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/SR/test_sr.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, model_optimizer, unet_config = config['Unet'].values()
    model_cls = get_obj_from_str(model_cls_str)
    gaussian_config = config['Gaussian']

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)
    input_shapes = (input_shape, input_shape[0])

    diffusion = GaussianSR(**gaussian_config, image_size=dataloader_configs['image_size'])

    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=unet_config)

    model_ckpt = {'model': state, 'steps': 0}
    model_save_path = trainer_configs['model_path']
    checkpoint_manager = create_checkpoint_manager(model_save_path, max_to_keep=1)
    if len(os.listdir(model_save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    dl = generator(**dataloader_configs)  # file_path
    finished_steps = model_ckpt['steps']

    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, trainer_configs['total_steps']):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            batch = shard(batch)
            state, metrics = train_step(state, batch, train_step_key, diffusion)
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            pbar.set_postfix(metrics)
            pbar.update(1)

            if steps > 100:
                state = update_ema(state, 0.9999)

            if steps % trainer_configs['sample_steps'] == 0:
                batch = einops.rearrange(batch, 'n b h w c -> (n b ) h w c')
                try:
                    sample_save_image_sr(key, diffusion, steps, state, batch, save_path=trainer_configs['save_path'])
                except Exception as e:
                    print(e)
                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)


if __name__ == "__main__":
    train()
