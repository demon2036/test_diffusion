import argparse
from tqdm import tqdm
import jax.random
from data.dataset import generator
from modules.gaussian.gaussianDecoder import GaussianDecoder
from modules.state_utils import create_state
from modules.utils import EMATrainState, create_checkpoint_manager, load_ckpt, read_yaml, update_ema, \
    sample_save_image_diffusion, get_obj_from_str, sample_save_image_diffusion_encoder
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

@partial(jax.pmap, static_broadcasted_argnums=(3), axis_name='batch')
def train_step(state, batch, train_key, cls):
    def loss_fn(params):
        p_loss,kl_loss = cls(train_key, state, params, batch)
        return p_loss+kl_loss*1e-6,(p_loss,kl_loss)

    grad_fn = jax.value_and_grad(loss_fn,has_aux=True)
    (loss,(p_loss,kl_loss)), grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"p_loss": loss,'kl_loss':kl_loss}
    return new_state, metric




def copy_params_to_ema(state):
   state = state.replace(ema_params = state.params)
   return state

def apply_ema_decay(state, ema_decay):
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay + p * (1. - ema_decay), state.ema_params, state.params)
    state = state.replace(ema_params = params_ema)
    return state

def ema_decay_schedule(step):
    beta = 0.995
    update_every = 10
    update_after_step = 100
    inv_gamma = 1.0
    power = 2 / 3
    min_value = 0.0

    count = jnp.clip(step - update_after_step - 1, a_min=0.)
    value = 1 - (1 + count / inv_gamma) ** - power
    ema_rate = jnp.clip(value, a_min=min_value, a_max=beta)
    return ema_rate









def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='configs/DiffusionEncoder/test_diff.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str,model_optimizer, unet_config = config['Unet'].values()
    model_cls = get_obj_from_str(model_cls_str)
    gaussian_config = config['Gaussian']

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)
    input_shapes = (input_shape, input_shape[0],input_shape)

    c = GaussianDecoder(**gaussian_config, image_size=dataloader_configs['image_size'])

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

    p_copy_params_to_ema=jax.pmap(copy_params_to_ema)
    p_apply_ema=jax.pmap(apply_ema_decay)

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

            if steps <= 100:
                state = p_copy_params_to_ema(state)
            elif steps % 10 == 0:
                ema_decay = ema_decay_schedule(steps)

                state = p_apply_ema(state, shard(jnp.array([ema_decay])))


            # if steps > 100 and steps % 10 == 0:
            #     state = update_ema(state, 0.995)

            if steps % trainer_configs['sample_steps'] == 0:
                batch=flax.jax_utils.unreplicate(batch)
                try:
                    sample_save_image_diffusion_encoder(key, c, steps, state, trainer_configs['save_path'],batch)
                except Exception as e:
                    print(e)

                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)


if __name__ == "__main__":
    train()
