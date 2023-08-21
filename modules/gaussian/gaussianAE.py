from modules.gaussian.gaussian import Gaussian, extract, ModelPrediction
import jax
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp


@jax.pmap
def model_predict(model, x, time):
    return model.apply_fn({"params": model.ema_params}, x, time)


class GaussianAE(Gaussian):
    def __init__(
            self,
            *args,
            **kwargs

    ):
        super().__init__(*args, **kwargs)

    def __call__(self, key, img):
        key_times, key_noise = jax.random.split(key, 2)
        b, h, w, c = img.shape
        t = jax.random.randint(key_times, (b,), minval=0, maxval=self.num_timesteps)
        noise = self.generate_nosie(key_noise, img.shape)

        return self.q_sample(img, t, noise)
