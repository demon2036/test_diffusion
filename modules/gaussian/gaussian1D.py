from collections import namedtuple
from functools import partial
import numpy as np
from einops import einops
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm

from modules.gaussian.gaussian import Gaussian
from modules.noise.noise import normal_noise, truncate_noise, pyramid_nosie, resize_noise, offset_noise
from modules.gaussian.schedules import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from modules.loss.loss import l1_loss, l2_loss
import jax
import jax.numpy as jnp

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))






class Gaussian1D(Gaussian):
    def __init__(
            self,
            latent_size,
            *args,
            **kwargs

    ):
        super().__init__(*args,**kwargs)
        self.latent_size = latent_size








    def ddim_sample(self, key, state, self_condition=None, shape=None):
        b, *_ = shape
        key, key_image = jax.random.split(key, 2)
        img = self.generate_nosie(key_image, shape=shape)

        times = np.asarray(np.linspace(-1, 999, num=self.sampling_timesteps + 1), dtype=np.int32)
        times = list(reversed(times))

        img = shard(img)

        x_self_cond = self_condition
        has_condition = False
        if x_self_cond is not None:
            x_self_cond = shard(x_self_cond)
            has_condition = True

        x_start = jnp.zeros_like(img)
        for time, time_next in tqdm(zip(times[:-1], times[1:]), total=self.sampling_timesteps):
            batch_times = jnp.full((b,), time)

            if has_condition:
                pass
            elif self.self_condition:
                x_self_cond = x_start
            else:
                x_self_cond = None

            batch_times = shard(batch_times)

            pred_noise, x_start = self.pmap_model_predictions(x=img, t=batch_times, x_self_cond=x_self_cond,
                                                              state=state)

            if time_next < 0:
                img = x_start
            else:
                key, key_noise = jax.random.split(key, 2)
                noise = pred_noise

                # if time_next > 100:
                #     noise = self.generate_nosie(key_noise, shape=shape)
                # else:
                #     noise = pred_noise

                batch_times_next = jnp.full((b,), time_next)
                batch_times_next = shard(batch_times_next)
                img = self.pmap_q_sample(x_start, batch_times_next, noise)

        img = einops.rearrange(img, 'n b  c->(n b ) c', n=img.shape[0])

        return img

    def sample(self, key, state, self_condition=None, batch_size=64):

        if self_condition is not None:
            batch_size = self_condition.shape[0]

        shape = (batch_size, self.latent_size)

        if self.num_timesteps > self.sampling_timesteps:
            samples = self.ddim_sample(key, state, self_condition, shape)
        else:
            samples = self.p_sample_loop(key, state, self_condition, shape)

        return samples / self.scale_factor

