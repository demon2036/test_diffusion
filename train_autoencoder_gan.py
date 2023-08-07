from flax.training import orbax_utils
from flax.training.common_utils import shard_prng_key, shard
from data.dataset import generator
from modules.models.autoencoder import AutoEncoder
from functools import partial
import jax
import jax.numpy as jnp
from modules.loss.loss import l1_loss, l2_loss, hinge_d_loss
import optax
import argparse

from modules.state_utils import create_state
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt, update_ema, sample_save_image_autoencoder, \
    get_obj_from_str, EMATrainState
import os
import flax
from tqdm import tqdm

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def kl_divergence(mean, logvar):
    return 0.5 * jnp.sum(jnp.power(mean, 2) + jnp.exp(logvar) - 1.0 - logvar, axis=[1, 2, 3])


def adoptive_weight(disc_start, discriminator_state, reconstruct):
    if disc_start:
        fake_logit, _ = discriminator_state.apply_fn(
            {'params': discriminator_state.params, 'batch_stats': discriminator_state.batch_stats}, reconstruct,
            mutable=['batch_stats'])
        return -fake_logit.mean()
    else:
        return 0


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))  # static_broadcasted_argnums=(3),
def train_step(state: EMATrainState, x, discriminator_state: EMATrainState, test: bool, z_rng):
    def loss_fn(params):
        reconstruct, intermediate = state.apply_fn({'params': params}, x, z_rng=z_rng, mutable=['intermediate'])

        z_mean = intermediate['intermediate']['mean'][0]
        z_variance = intermediate['intermediate']['variance'][0]
        kl_loss = kl_divergence(z_mean, z_variance).mean()

        gan_loss = adoptive_weight(test, discriminator_state, reconstruct)
        rec_loss = l1_loss(reconstruct, x).mean()
        return rec_loss + 0.5 * gan_loss + 1e-6 * kl_loss, (rec_loss, gan_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (rec_loss, gan_loss, kl_loss)), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads)
    rec_loss = jax.lax.pmean(rec_loss, axis_name='batch')
    gan_loss = jax.lax.pmean(gan_loss, axis_name='batch')
    metric = {"rec_loss": rec_loss, 'gan_loss': gan_loss, 'kl_loss': kl_loss}
    return new_state, metric


@partial(jax.pmap, axis_name='batch')  # static_broadcasted_argnums=(3),
def train_step_disc(state: EMATrainState, x, discriminator_state: EMATrainState, z_rng):
    def loss_fn(params):
        fake_image = state.apply_fn({'params': state.params}, x, z_rng=z_rng)
        real_image = x

        logit_real, mutable = discriminator_state.apply_fn(
            {'params': params, 'batch_stats': discriminator_state.batch_stats}, real_image, True,
            mutable=['batch_stats'])

        logit_fake, mutable = discriminator_state.apply_fn(
            {'params': params, 'batch_stats': mutable['batch_stats']}, fake_image, True,
            mutable=['batch_stats'])

        disc_loss = hinge_d_loss(logit_real, logit_fake)
        return disc_loss, mutable

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (disc_loss, mutable), grads = grad_fn(discriminator_state.params, )
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_disc_state = discriminator_state.apply_gradients(grads=grads, batch_stats=mutable['batch_stats'])
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
                         optimizer_dict=model_optimizer,batch_size=dataloader_configs['batch_size'],
                         train_state=EMATrainState, model_kwargs=model_configs)

    discriminator_state = create_state(rng=key, model_cls=disc_cls, input_shapes=input_shapes,
                                       optimizer_dict=disc_optimizer,batch_size=dataloader_configs['batch_size'],
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

    disc_start = finished_steps >= trainer_configs['disc_start']
    metrics = {}
    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, 1000000):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            batch = shard(batch)
            state, metrics = train_step(state, batch, discriminator_state, disc_start, train_step_key)
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            if steps == trainer_configs['disc_start']:
                disc_start = True

            if steps > trainer_configs['disc_start']:
                discriminator_state, metrics_disc = train_step_disc(state, batch, discriminator_state, train_step_key)
                for k, v in metrics_disc.items():
                    metrics_disc.update({k: v[0]})
                metrics.update(metrics_disc)
            pbar.set_postfix(metrics)
            pbar.update(1)

            # if steps > 100:
            #     state = update_ema(state, 0.9999)
            if steps > 0 and steps % 1 == 0:
                decay = min(0.9999, (1 + steps) / (10 + steps))
                decay = flax.jax_utils.replicate(jnp.array([decay]))
                state = update_ema(state, decay)

            if steps % trainer_configs['sample_steps'] == 0:
                save_path = f"{trainer_configs['save_path']}"
                sample_save_image_autoencoder(state, save_path, steps, batch,key)
                un_replicate_state = flax.jax_utils.unreplicate(state)
                un_replicate_disc_state = flax.jax_utils.unreplicate(discriminator_state)
                model_ckpt = {'model': un_replicate_state, 'discriminator': un_replicate_disc_state, 'steps': steps}
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args},
                                        force=False)
