from functools import partial

from modules.gaussian.gaussian import Gaussian, extract, ModelPrediction, identity
import jax
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp

from modules.utils import default, get_obj_from_str


def kl_divergence(mean, logvar):
    return 0.5 * jnp.sum(jnp.power(mean, 2) + jnp.exp(logvar) - 1.0 - logvar, axis=[1, 2, 3])


def model_predict_ema(model, x, time, x_self_cond=None, method=None):
    print(f'method:{method}')
    return model.apply_fn({"params": model.ema_params}, x, time, x_self_cond, method=method)


def model_predict(model, x, time, x_self_cond=None, method=None):
    return model.apply_fn({"params": model.params}, x, time, x_self_cond, method=method)


class GaussianDecoder(Gaussian):
    def __init__(
            self,
            apply_method=None,
            kl_loss=0,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        if apply_method is not None:
            if not callable(apply_method):
                apply_method = get_obj_from_str(apply_method)
        self.kl_loss = kl_loss
        self.apply_method = apply_method
        print(f'self.apply_method:{self.apply_method}')

    def model_predictions(self, x, t=None, x_self_cond=None, state=None, rederive_pred_noise=False, *args, **kwargs):
        if self.train_state:
            model_output = model_predict(state, x, t, x_self_cond, self.apply_method)
        else:
            model_output = model_predict_ema(state, x, t, x_self_cond, self.apply_method)

        clip_x_start = self.clip_x_start
        maybe_clip = partial(jnp.clip, a_min=-1., a_max=1.) if clip_x_start else identity

        if self.objective == 'predict_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'predict_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'predict_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'predict_mx':
            x_start = -model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

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

    def __call__(self, key, state, params, img):
        key_times, key_noise = jax.random.split(key, 2)
        b, h, w, c = img.shape
        t = jax.random.randint(key_times, (b,), minval=0, maxval=self.num_timesteps)

        return self.p_loss(key_noise, state, params, img, t)
