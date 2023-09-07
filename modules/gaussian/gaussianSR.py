import einops
from flax.training.common_utils import shard, shard_prng_key

from modules.gaussian.gaussian import Gaussian, extract, ModelPrediction
import jax
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp


class GaussianSR(Gaussian):
    def __init__(
            self,
            sr_factor,
            predict_residual=True,
            *args,
            **kwargs

    ):
        super().__init__(*args, **kwargs)
        self.sr_factor = sr_factor
        self.predict_residual = predict_residual

    def p_sample_loop(self, key, state, x_self_cond=None, shape=None):
        key, normal_key = jax.random.split(key, 2)
        img = self.generate_nosie(normal_key, shape)
        img = shard(img)

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            key, normal_key = jax.random.split(key, 2)

            normal_key = shard_prng_key(normal_key)
            t = shard(t)
            img, x_start = self.p_sample(normal_key, img, t, x_self_cond)

        ret = einops.rearrange(img, 'n b h w c->(n b ) h w c')

        return ret

    def sample(self, key, state, lr_image):

        b, h, w, c = lr_image.shape
        lr_image = jax.image.resize(lr_image, (b, h * self.sr_factor, w * self.sr_factor, c), method='bicubic')
        noise_shape = lr_image.shape

        if self.num_timesteps > self.sampling_timesteps:
            res = self.ddim_sample(key, state, lr_image, noise_shape)
        else:
            res = self.p_sample_loop(key, state, lr_image, noise_shape)
        # res = self.ddim_sample(key, state, lr_image, noise_shape)
        if self.predict_residual:
            ret = res + lr_image

        return [ret, res, lr_image]

    def p_loss(self, key, state, params, x_start, t):
        noise = self.generate_nosie(key, shape=x_start.shape)

        b, h, w, c = x_start.shape
        lr_image = jax.image.resize(x_start, shape=(b, h // self.sr_factor, w // self.sr_factor, c), method='bilinear')
        fake_image = jax.image.resize(lr_image, shape=(b, h, w, c), method='bicubic')

        if self.predict_residual:
            x_start = x_start - fake_image

        x = self.q_sample(x_start, t, noise)
        assert x_start.shape[1:] == tuple(self.sample_shape)
        model_output = state.apply_fn({"params": params}, x, t, fake_image)

        if self.objective == 'predict_noise':
            target = noise
        elif self.objective == 'predict_x0':
            target = x_start
        elif self.objective == 'predict_v':
            v = self.predict_v(x_start, t, noise)
            target = v
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
