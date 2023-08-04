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
    get_obj_from_str, EMATrainState, sample_save_image_diffusion, sample_save_image_latent_diffusion
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




def get_auto_encoder(config):
    model_cls_str, model_optimizer, model_configs = config['AutoEncoder'].values()
    model_cls = get_obj_from_str(model_cls_str)
    disc_cls_str, disc_optimizer, disc_configs = config['Discriminator'].values()
    disc_cls = get_obj_from_str(disc_cls_str)

    key = jax.random.PRNGKey(seed=43)
    input_shape = (1, 256, 256, 3)
    input_shapes = (input_shape,)
    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=model_configs)

    discriminator_state = create_state(rng=key, model_cls=disc_cls, input_shapes=input_shapes,
                                       optimizer_dict=disc_optimizer,
                                       train_state=EMATrainState, model_kwargs=disc_configs)

    model_ckpt = {'model': state, 'discriminator': discriminator_state, 'steps': 0}
    save_path = '/home/john/pythonfile/diffusion/model/ldm/check_points/AutoEncoder'
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=1)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/training/ldm/test_diff.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, model_optimizer, unet_config = config['Unet'].values()
    model_cls = get_obj_from_str(model_cls_str)
    gaussian_config = config['Gaussian']
    first_stage_config=config['FirstStage']

    ae_state = get_auto_encoder(first_stage_config)
    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)
    input_shapes = (input_shape, input_shape[0])

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

    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, 1000000):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            print(batch.shape,batch.min(),batch.max())

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
                    sample_save_image_latent_diffusion(key, c, steps, state, trainer_configs['save_path'],ae_state)
                except Exception as e:
                    print(e)

                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)
