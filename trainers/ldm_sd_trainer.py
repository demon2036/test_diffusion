import einops
from diffusers import FlaxAutoencoderKL
from flax.jax_utils import replicate
from flax.training import orbax_utils
from flax.training.common_utils import shard_prng_key, shard
from functools import partial
import jax
import jax.numpy as jnp

from modules.infer_utils import sample_save_image_latent_diffusion, sample_save_image_latent_diffusion_sd
from modules.models.diffEncoder import DiffEncoder
from modules.utils import create_checkpoint_manager, load_ckpt, update_ema, default
import os
import flax
from tqdm import tqdm

from trainers.basic_trainer import Trainer


# vae, params = FlaxAutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', from_pt=True, subfolder='vae',
#                                                 # cache_dir='sd'
#                                                 )


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


@partial(jax.pmap, static_broadcasted_argnums=(3, 4), axis_name='batch')
def train_step_with_encode(state, batch, train_key, cls, dummy_vae):
    train_key, z_rng = jax.random.split(train_key, 2)
    data = einops.rearrange(batch, 'b h w c->b c h w ')

    def loss_fn(params):
        posterior = dummy_vae.vae.apply({'params': dummy_vae.vae_params}, data, method=FlaxAutoencoderKL.encode)
        latent = posterior.latent_dist.sample(z_rng)
        loss = cls(train_key, state, params, latent)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"loss": loss}
    return new_state, metric


class Dummy:
    def __init__(self, vae, vae_params):
        self.vae = vae
        self.vae_params = vae_params


class LdmSDTrainer(Trainer):
    def __init__(self,
                 state,
                 gaussian,
                 vae,
                 vae_params,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.state = state
        self.gaussian = gaussian
        self.vae_dummy = Dummy(vae, vae_params)
        # self.vae_params = vae_params
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

        try:
            sample_save_image_latent_diffusion_sd(self.rng,
                                                  self.gaussian,
                                                  self.finished_steps,
                                                  sample_state,
                                                  self.save_path,
                                                  self.vae_dummy)
        except Exception as e:
            print(e)

    def train(self):
        state = flax.jax_utils.replicate(self.state)
        self.finished_steps += 1
        with tqdm(total=self.total_steps) as pbar:
            pbar.update(self.finished_steps)
            while self.finished_steps < self.total_steps:
                self.rng, train_step_key = jax.random.split(self.rng, num=2)
                train_step_key = shard_prng_key(train_step_key)
                batch = next(self.dl)
                # batch = shard(batch)

                if self.data_type == 'np':
                    state, metrics = train_step(state, batch, train_step_key, self.gaussian)
                elif self.data_type == 'img':
                    state, metrics = train_step_with_encode(state, batch, train_step_key, self.gaussian,
                                                            self.vae_dummy,
                                                            )
                else:
                    raise NotImplemented()

                for k, v in metrics.items():
                    metrics.update({k: v[0]})

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
