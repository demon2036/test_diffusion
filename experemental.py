import einops
from flax.training import orbax_utils
from flax.training.common_utils import shard_prng_key, shard
from data.dataset import generator, get_dataloader, torch_to_jax
from modules.gaussian.gaussian import Gaussian
from modules.models.autoencoder import AutoEncoder
from functools import partial
import jax
import jax.numpy as jnp
from modules.loss.loss import l1_loss, l2_loss, hinge_d_loss
import optax
import argparse
from tools.resize_dataset import save_image

from modules.save_utils import save_image_from_jax
from modules.state_utils import create_state
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt, update_ema, sample_save_image_autoencoder, \
    get_obj_from_str, EMATrainState, sample_save_image_diffusion, sample_save_image_latent_diffusion, \
    sample_save_image_latent_diffusion, sample_save_image_latent_diffusion_1d_test, \
    sample_save_image_latent_diffusion_1d_test2
import os
import flax
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


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


def get_auto_encoder_diff(config):
    ae_cls_str, model_optimizer, model_configs = config['AutoEncoder'].values()
    gaussian, gaussian_configs = get_obj_from_str(config['Gaussian']['target']), config['Gaussian']['params']

    print(config['Gaussian']['target'])

    first_stage_gaussian = gaussian(**gaussian_configs)
    ae_cls = get_obj_from_str(ae_cls_str)

    key = jax.random.PRNGKey(seed=43)
    input_shape = (1, 256, 256, 3)
    input_shapes = (input_shape, input_shape[0], input_shape)
    state = create_state(rng=key, model_cls=ae_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=model_configs)

    model_ckpt = {'model': state, 'steps': 0}
    save_path = './model/DiffAE'
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=1)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    return state, first_stage_gaussian


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/training/ldm_1d/test_diff.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, model_optimizer, unet_config = config['LatentNet'].values()
    model_cls = get_obj_from_str(model_cls_str)

    gaussian, gaussian_configs = get_obj_from_str(config['Gaussian']['target']), config['Gaussian']['params']

    first_stage_config = config['FirstStage']

    ae_state, first_stage_gaussian = get_auto_encoder_diff(first_stage_config)

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    input_shape = (1, 512)
    input_shapes = (input_shape, input_shape[0])

    c = gaussian(**gaussian_configs, )

    print(model_cls)

    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=unet_config)

    model_ckpt = {'model': state, 'steps': 0}
    model_save_path = trainer_configs['model_path']

    checkpoint_manager = create_checkpoint_manager(model_save_path, max_to_keep=5)
    if len(os.listdir(model_save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])

    print(dataloader_configs)

    dl = generator(**dataloader_configs)  # file_path
    finished_steps = model_ckpt['steps']

    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, trainer_configs['total_steps'] + 1):

            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)
            #print(f'batch:{batch.shape}')
            batch = shard(batch)
            #print(f'batch:{batch.shape}')


            # sample_save_image_latent_diffusion_1d_test(key, c, steps, state, trainer_configs['save_path'], ae_state,
            #                                            first_stage_gaussian, batch)
            state, metrics = train_step(state, batch, train_step_key, c)
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            pbar.set_postfix(metrics)
            pbar.update(1)

            if steps > 0 and steps % 1 == 0:
                decay = min(0.9999, (1 + steps) / (10 + steps))
                decay = flax.jax_utils.replicate(jnp.array([decay]))
                state = update_ema(state, decay)

            if steps % trainer_configs['sample_steps'] == 0:
                try:
                    sample_save_image_latent_diffusion(key, c, steps, state, trainer_configs['save_path'], ae_state,first_stage_gaussian,)
                    # sample_save_image_latent_diffusion_1d_test2(key, c, steps, state, trainer_configs['save_path'], ae_state,
                    #                                       first_stage_gaussian, batch)

                except Exception as e:
                    print(e)
                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)
