from modules.gaussian.gaussian import Gaussian, extract, ModelPrediction
import jax
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp


class GaussianDecoder(Gaussian):
    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

    def p_loss(self, key, state, params, x_start, t):
        key,z_rng=jax.random.split(key,2)
        noise = self.generate_nosie(key, shape=x_start.shape)
        assert x_start.shape[1:] == tuple(self.sample_shape)

        x = self.q_sample(x_start, t, noise)
        model_output = state.apply_fn({"params": params}, x, t, x_start,z_rng=z_rng)

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
