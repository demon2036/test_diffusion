from functools import partial

import jax
import jax.numpy as jnp
from einops import reduce, rearrange

from modules.score.elucidate import ElucidatedDiffusion
from modules.utils import get_obj_from_str


def kl_divergence(mean, logvar,mean_weight=1.0,std_weight=1.0):
    return 0.5 * jnp.sum(mean_weight*jnp.power(mean, 2) + std_weight*(jnp.exp(logvar) - 1.0 - logvar), axis=[1, 2, 3])


def model_predict_ema(model, x, time, x_self_cond=None, method=None):
    print(f'method:{method}')
    return model.apply_fn({"params": model.ema_params}, x, time, x_self_cond, method=method)


def model_predict(model, x, time, x_self_cond=None, method=None):
    return model.apply_fn({"params": model.params}, x, time, x_self_cond, method=method)


class ElucidateAutoEncoder(ElucidatedDiffusion):
    def __init__(
            self,
            kl_loss=0,
            mean_weight=1.0,
            std_weight=1.0,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_weight=mean_weight
        self.std_weight=std_weight
        self.kl_loss = kl_loss

    def p_loss(self, key, state, params, images, x_self_cond=None):

        key_noise, key_sigmas, key_noise,train_key = jax.random.split(key, 4)

        batch_size = images.shape[0]

        assert images.shape[1:] == tuple(self.sample_shape)

        sigmas = self.noise_distribution(key_sigmas, batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = self.generate_noise(key_noise, images.shape)

        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        denoised, mod_vars = self.preconditioned_network_forward(noised_images, sigmas, x_self_cond,
                                                                 state=state,
                                                                 params=params,
                                                                 return_mod_vars=True,
                                                                 z_rng=train_key if self.train_state else None)

        if self.kl_loss > 0:
            mean = mod_vars['intermediates']['mean'][0]
            log_var = mod_vars['intermediates']['log_var'][0]
            kl_loss = kl_divergence(mean, log_var,self.mean_weight,self.std_weight) * self.kl_loss
        else:
            kl_loss = jnp.array([0])

        losses = self.loss(denoised, images)
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean(), kl_loss.mean()

    """
    def p_loss(self, key, state, params, x_start, t):
        key, z_rng = jax.random.split(key, 2)
        noise = self.generate_nosie(key, shape=x_start.shape)

        assert x_start.shape[1:] == tuple(self.sample_shape)

        x = self.q_sample(x_start, t, noise)
        model_output, mod_vars = state.apply_fn({"params": params}, x, t, x_start, z_rng=z_rng, mutable='intermediates')
        if self.kl_loss > 0:
            mean = mod_vars['intermediates']['mean'][0]
            log_var = mod_vars['intermediates']['log_var'][0]
            kl_loss = kl_divergence(mean, log_var)*self.kl_loss
        else:
            kl_loss = jnp.array([0])

        if self.objective == 'predict_noise':
            target = noise
        elif self.objective == 'predict_x0':
            target = x_start
        elif self.objective == 'predict_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        elif self.objective == 'predict_mx':
            target = -x_start
        else:
            raise NotImplemented()

        p_loss = self.loss(target, model_output)

        p_loss = (p_loss * extract(self.loss_weight, t, p_loss.shape)).mean()
        return p_loss,kl_loss.mean()
    """

    def __call__(self, key, state, params, images):
        return self.p_loss(key, state, params, images,images)