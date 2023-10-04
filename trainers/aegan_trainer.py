import einops
import optax
from tqdm import tqdm
import jax.random
from flax.training.train_state import TrainState
from modules.infer_utils import sample_save_image_diffusion_encoder, sample_save_image_autoencoder
from modules.loss.loss import hinge_d_loss, l1_loss
from modules.state_utils import EMATrainState
from modules.utils import create_checkpoint_manager, load_ckpt, update_ema, \
    get_obj_from_str, default
import flax
import os
from functools import partial
from flax.training import orbax_utils
from flax.training.common_utils import shard, shard_prng_key
import jax.numpy as jnp
from modules.augments import get_cut_mix_label, get_mix_up_label
from trainers.basic_trainer import Trainer

""""""


def adoptive_weight(disc_start, discriminator_state, reconstruct):
    if disc_start:

        variable = {'params': discriminator_state.params}
        if discriminator_state.batch_stats is not None:
            variable.update({'batch_stats': discriminator_state.batch_stats})

        fake_logit, _ = discriminator_state.apply_fn(variable, reconstruct, mutable=['batch_stats'])

        return optax.l2_loss(fake_logit, jnp.ones_like(fake_logit)).mean()
        # return -fake_logit.mean()
    else:
        return 0


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))  # static_broadcasted_argnums=(3),
def train_step(state: EMATrainState, x, discriminator_state: EMATrainState, test: bool, z_rng):
    def loss_fn(params):
        reconstruct = state.apply_fn({'params': params}, x, z_rng=z_rng, )
        # reconstruct, intermediate = state.apply_fn({'params': params}, x, z_rng=z_rng, mutable=['intermediate'])

        # z_mean = intermediate['intermediate']['mean'][0]
        # z_variance = intermediate['intermediate']['variance'][0]
        # kl_loss = kl_divergence(z_mean, z_variance).mean()
        # + 1e-6 * kl_loss
        kl_loss = 0
        gan_loss = adoptive_weight(test, discriminator_state, reconstruct)
        rec_loss = l1_loss(reconstruct, x).mean()
        return rec_loss + 0.05 * gan_loss + 1e-6 * kl_loss, (rec_loss, gan_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (rec_loss, gan_loss, kl_loss)), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads)
    rec_loss = jax.lax.pmean(rec_loss, axis_name='batch')
    gan_loss = jax.lax.pmean(gan_loss, axis_name='batch')
    metric = {"rec_loss": rec_loss, 'gan_loss': gan_loss, 'kl_loss': kl_loss}
    return new_state, metric


@partial(jax.pmap, axis_name='batch')
def train_step_disc(state: EMATrainState, x, discriminator_state: EMATrainState, z_rng):
    def loss_fn(params):
        fake_image = state.apply_fn({'params': state.params}, x, z_rng=z_rng)
        real_image = x

        variable = {'params': discriminator_state.params}
        if discriminator_state.batch_stats is not None:
            variable.update({'batch_stats': params})

        logit_real, mutable = discriminator_state.apply_fn(variable, real_image, True,
                                                           mutable=['batch_stats'])

        logit_fake, mutable = discriminator_state.apply_fn(variable, fake_image, True,
                                                           mutable=['batch_stats'])

        fake_loss = optax.l2_loss(logit_fake, jnp.zeros_like(logit_fake))
        real_loss = optax.l2_loss(logit_real, jnp.ones_like(logit_real))
        mix_label_p = get_cut_mix_label(x, z_rng)
        mixed_image = mix_label_p * real_image + (1 - mix_label_p) * fake_image
        logit_mixed, mutable = discriminator_state.apply_fn(variable, mixed_image, True,
                                                            mutable=['batch_stats'])
        loss_cut_mix = optax.l2_loss(mix_label_p, logit_mixed)

        mix_up_label = get_mix_up_label(x.shape, z_rng)
        mixed_image = mix_up_label * real_image + (1 - mix_up_label) * fake_image
        logit_mixed, mutable = discriminator_state.apply_fn(variable, mixed_image, True,
                                                            mutable=['batch_stats'])
        loss_mixe_up = optax.l2_loss(mix_up_label, logit_mixed)

        disc_loss = (fake_loss.mean() + real_loss.mean() + loss_mixe_up.mean() + loss_cut_mix.mean()).mean()
        # disc_loss = hinge_d_loss(logit_real, logit_fake)
        return disc_loss, mutable

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (disc_loss, mutable), grads = grad_fn(discriminator_state.params, )
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_disc_state = discriminator_state.apply_gradients(grads=grads, batch_stats=mutable[
        'batch_stats'] if 'batch_stats' in mutable else None)
    disc_loss = jax.lax.pmean(disc_loss, axis_name='batch')
    metric = {'disc_loss': disc_loss}
    return new_disc_state, metric


class AutoEncoderTrainer(Trainer):
    def __init__(self,
                 state,
                 disc_state,
                 disc_start,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.state = state
        self.disc_state = disc_state
        self.disc_start = disc_start
        self.template_ckpt = {'model': self.state, 'steps': self.finished_steps}

    def load(self, model_path=None, template_ckpt=None):

        if model_path is not None:
            checkpoint_manager = create_checkpoint_manager(model_path, max_to_keep=1)
        else:
            checkpoint_manager = self.checkpoint_manager

        model_ckpt = default(template_ckpt, self.template_ckpt)
        if len(os.listdir(self.model_path)) > 0:
            model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)
        self.state = model_ckpt['model']
        self.finished_steps = model_ckpt['steps']

    def save(self):
        model_ckpt = {'model': self.state, 'steps': self.finished_steps}
        save_args = orbax_utils.save_args_from_target(model_ckpt)
        self.checkpoint_manager.save(self.finished_steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)

    def sample(self, sample_state=None):
        sample_state = default(sample_state, flax.jax_utils.replicate(self.state))
        batch = next(self.dl)
        # batch = batch.reshape(-1, *batch.shape[2:])
        try:
            sample_save_image_autoencoder(sample_state,
                                          self.save_path,
                                          self.finished_steps,
                                          batch,
                                          self.rng
                                          )
        except Exception as e:
            print(e)

    def train(self):
        state = flax.jax_utils.replicate(self.state)
        discriminator_state = flax.jax_utils.replicate(self.disc_state)

        self.finished_steps += 1
        disc_start = self.finished_steps >= self.disc_start

        with tqdm(total=self.total_steps) as pbar:
            pbar.update(self.finished_steps)
            while self.finished_steps < self.total_steps:
                self.rng, train_step_key = jax.random.split(self.rng, num=2)
                train_step_key = shard_prng_key(train_step_key)
                batch = next(self.dl)
                # batch = shard(batch)


                state, metrics = train_step(state, batch, discriminator_state, disc_start, train_step_key)

                for k, v in metrics.items():
                    metrics.update({k: v[0]})

                if self.finished_steps == self.disc_start:
                    disc_start = True

                if self.finished_steps > self.disc_start:

                    discriminator_state, metrics_disc = train_step_disc(state, batch, discriminator_state,
                                                                        train_step_key, )
                    for k, v in metrics_disc.items():
                        metrics_disc.update({k: v[0]})
                    metrics.update(metrics_disc)
                    """"""

                pbar.set_postfix(metrics)
                pbar.update(1)

                if self.finished_steps > 0 and self.finished_steps % 1 == 0:
                    decay = min(0.9999, (1 + self.finished_steps) / (10 + self.finished_steps))
                    decay = flax.jax_utils.replicate(jnp.array([decay]))
                    state = update_ema(state, decay)

                if self.finished_steps % self.sample_steps == 0:
                    print(self.finished_steps, self.sample_steps)
                    self.sample(state)
                    self.state = flax.jax_utils.unreplicate(state)
                    self.save()

                self.finished_steps += 1
