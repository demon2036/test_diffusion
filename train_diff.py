import argparse

from flax.jax_utils import replicate
from tqdm import tqdm
import jax.random
from data.dataset import generator
from modules.state_utils import create_state, apply_ema_decay, copy_params_to_ema, ema_decay_schedule
from modules.utils import EMATrainState, create_checkpoint_manager, load_ckpt, read_yaml, update_ema, \
    sample_save_image_diffusion, get_obj_from_str
import flax
import os
from functools import partial
from flax.training import orbax_utils
from flax.training.common_utils import shard, shard_prng_key
from jax_smi import initialise_tracking
from modules.gaussian.gaussian import Gaussian
import jax.numpy as jnp

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'




def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='configs/training/Diffusion/test.yaml')
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
    input_shapes = (input_shape, input_shape[0], input_shape)

    c = Gaussian(**gaussian_config, image_size=dataloader_configs['image_size'])

    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=unet_config)

    model_ckpt = {'model': state, 'steps': 0}
    model_save_path = trainer_configs['model_path']

    checkpoint_manager = create_checkpoint_manager(model_save_path, max_to_keep=5)
    if len(os.listdir(model_save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    dl = generator(**dataloader_configs)  # file_path
    finished_steps = model_ckpt['steps']

    p_copy_params_to_ema = jax.pmap(copy_params_to_ema)
    p_apply_ema = jax.pmap(apply_ema_decay)

    num_repeats = 1

    with tqdm(total=trainer_configs['total_steps'] * num_repeats) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, 1000000):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            batch = shard(batch)
            for _ in range(num_repeats):
                state, metrics = train_step(state, batch, train_step_key, c)
                for k, v in metrics.items():
                    metrics.update({k: v[0]})

                pbar.set_postfix(metrics)
                pbar.update(1)

            # if steps > 100 and steps % 1 == 0:
            #     state = update_ema(state, 0.9999)
            if steps <= 100:
                state = p_copy_params_to_ema(state)
            elif steps % 10 == 0:
                ema_decay = ema_decay_schedule(steps)
                state = p_apply_ema(state, replicate(jnp.array([ema_decay])))

            # if steps > 0 and steps % 1 == 0:
            #     decay = min(0.9999, (1 + steps) / (10 + steps))
            #     decay = flax.jax_utils.replicate(jnp.array([decay]))
            #     state = update_ema(state, decay)

            if steps % trainer_configs['sample_steps'] == 0:
                # sample_save_image_diffusion(key, c, steps, state, trainer_configs['save_path'])
                try:
                    sample_save_image_diffusion(key, c, steps, state, trainer_configs['save_path'])
                except Exception as e:
                    print(e)

                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)


if __name__ == "__main__":
    train()
