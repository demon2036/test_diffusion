from functools import partial

from modules.gaussian.gaussian import Gaussian, extract, ModelPrediction
import jax
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp

from modules.utils import default


def model_predict_ema(model, x, time, x_self_cond=None, method=None):
    print(f'method:{method}')
    return model.apply_fn({"params": model.ema_params}, x, time, x_self_cond, method=method)


def model_predict(model, x, time, x_self_cond=None, method=None):
    return model.apply_fn({"params": model.params}, x, time, x_self_cond, method=method)


class GaussianDecoder(Gaussian):
    def __init__(
            self,
            apply_method=None,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_method = default(apply_method, None)
        print(f'self.apply_method:{self.apply_method}')

    def model_predictions(self, x, t=None, x_self_cond=None, state=None, rederive_pred_noise=False, *args, **kwargs):
        # model_output = model_predict(state, x, t, x_self_cond)
        if self.train_state:
            model_output = model_predict(state, x, t, x_self_cond,self.apply_method)
        else:
            model_output = model_predict_ema(state, x, t, x_self_cond,self.apply_method)

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

    def model_predictions(self, x, t=None, x_self_cond=None, state=None, rederive_pred_noise=False, *args, **kwargs):
        # model_output = model_predict(state, x, t, x_self_cond)
        if self.train_state:
            model_output = model_predict(state, x, t, x_self_cond)
        else:
            model_output = model_predict_ema(state, x, t, x_self_cond, self.apply_fn)

        clip_x_start = True
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

        return ModelPrediction(pred_noise, x_start)

    def p_loss(self, key, state, params, x_start, t):
        key, z_rng = jax.random.split(key, 2)
        noise = self.generate_nosie(key, shape=x_start.shape)
        assert x_start.shape[1:] == tuple(self.sample_shape)

        x = self.q_sample(x_start, t, noise)
        model_output = state.apply_fn({"params": params}, x, t, x_start, z_rng=z_rng)

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
        return p_loss

    def __call__(self, key, state, params, img):
        key_times, key_noise = jax.random.split(key, 2)
        b, h, w, c = img.shape
        t = jax.random.randint(key_times, (b,), minval=0, maxval=self.num_timesteps)

        return self.p_loss(key_noise, state, params, img, t)
