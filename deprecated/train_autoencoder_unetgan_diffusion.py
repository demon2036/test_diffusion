from flax.training import orbax_utils
from flax.training.common_utils import shard_prng_key, shard
from data.dataset import generator
from modules.gaussian.gaussianAE import GaussianAE
from functools import partial
import jax
import jax.numpy as jnp
from modules.loss.loss import l1_loss, l2_loss, hinge_d_loss
import argparse

from modules.state_utils import create_state
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt, update_ema, sample_save_image_autoencoder, \
    get_obj_from_str, EMATrainState
import os
import flax
from tqdm import tqdm

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def adoptive_weight(disc_start, discriminator_state, reconstruct):
    if disc_start:
        fake_logit = discriminator_state.apply_fn(
            {'params': discriminator_state.params, }, reconstruct)
        return -fake_logit.mean()
    else:
        return 0


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,4,))  # static_broadcasted_argnums=(3),
def train_step_diffusion(state: EMATrainState, x, discriminator_state: EMATrainState, test: bool, diffusion: GaussianAE,
                         key):
    def loss_fn(params):
        key_real, key_fake = jax.random.split(key, 2)
        reconstruct = state.apply_fn({'params': params}, x)
        rec_loss = l1_loss(reconstruct, x).mean()

        reconstruct=diffusion(key_real,reconstruct)
        gan_loss = adoptive_weight(test, discriminator_state, reconstruct)

        return rec_loss + gan_loss * 0.1, (rec_loss, gan_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (rec_loss, gan_loss)), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads)
    rec_loss = jax.lax.pmean(rec_loss, axis_name='batch')
    gan_loss = jax.lax.pmean(gan_loss, axis_name='batch')
    metric = {"rec_loss": rec_loss, 'gan_loss': gan_loss}
    return new_state, metric


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))  # static_broadcasted_argnums=(3),
def train_step_disc_diffusion(state: EMATrainState, x, discriminator_state: EMATrainState, diffusion: GaussianAE, key):
    def loss_fn(params):
        key_real, key_fake = jax.random.split(key, 2)

        fake_image = state.apply_fn({'params': state.params}, x)
        real_image = x

        fake_image = diffusion(key_fake, fake_image)
        real_image = diffusion(key_fake, real_image)

        logit_real = discriminator_state.apply_fn({'params': params, }, real_image)
        logit_fake = discriminator_state.apply_fn({'params': params}, fake_image)

        disc_loss = hinge_d_loss(logit_real, logit_fake)
        return disc_loss, disc_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (disc_loss, _), grads = grad_fn(discriminator_state.params, )
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_disc_state = discriminator_state.apply_gradients(grads=grads)
    disc_loss = jax.lax.pmean(disc_loss, axis_name='batch')
    metric = {'disc_loss': disc_loss}
    return new_disc_state, metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/AutoEncoder/test_gan.yaml')

    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, model_optimizer, model_configs = config['AutoEncoder'].values()
    print(model_cls_str)
    model_cls = get_obj_from_str(model_cls_str)

    disc_cls_str, disc_optimizer, disc_configs = config['Discriminator'].values()
    disc_cls = get_obj_from_str(disc_cls_str)

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)
    input_shapes = (input_shape,)

    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=model_configs)

    discriminator_state = create_state(rng=key, model_cls=disc_cls, input_shapes=input_shapes,
                                       optimizer_dict=disc_optimizer,
                                       train_state=EMATrainState, model_kwargs=disc_configs)

    model_ckpt = {'model': state, 'discriminator': discriminator_state, 'steps': 0}
    save_path = trainer_configs['model_path']
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=1)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    discriminator_state = flax.jax_utils.replicate(model_ckpt['discriminator'])

    dl = generator(**dataloader_configs)  # file_path
    finished_steps = model_ckpt['steps']

    diffusion_model=GaussianAE()


    disc_start = finished_steps >= trainer_configs['disc_start']
    metrics = {}
    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, 1000000):
            key, train_step_key_generator,train_step_key_discriminator = jax.random.split(key, num=3)
            train_step_key_generator = shard_prng_key(train_step_key_generator)
            train_step_key_discriminator = shard_prng_key(train_step_key_discriminator)
            batch = next(dl)

            batch = shard(batch)
            state, metrics = train_step_diffusion(state, batch, discriminator_state, disc_start,diffusion_model,train_step_key_generator)
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            if steps == trainer_configs['disc_start']:
                disc_start = True

            if steps > trainer_configs['disc_start']:
                discriminator_state, metrics_disc = train_step_disc_diffusion(state, batch, discriminator_state,diffusion_model,train_step_key_discriminator)
                for k, v in metrics_disc.items():
                    metrics_disc.update({k: v[0]})
                metrics.update(metrics_disc)
            pbar.set_postfix(metrics)
            pbar.update(1)

            if steps > 100:
                state = update_ema(state, 0.9999)

            if steps % trainer_configs['sample_steps'] == 0:
                save_path = f"{trainer_configs['save_path']}"
                sample_save_image_autoencoder(state, save_path, steps, batch)
                un_replicate_state = flax.jax_utils.unreplicate(state)
                un_replicate_disc_state = flax.jax_utils.unreplicate(discriminator_state)
                model_ckpt = {'model': un_replicate_state, 'discriminator': un_replicate_disc_state, 'steps': steps}
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args},
                                        force=False)
