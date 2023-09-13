import einops
import jax
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm
import jax.numpy as jnp
from modules.score.elucidate_pmap import ElucidatedDiffusion
from modules.state_utils import EMATrainState
from modules.utils import extract


class ElucidateSR(ElucidatedDiffusion):
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

    # todo: implement sample function of scored-base diffusion sr

    def sample(self, key, state: EMATrainState, lr_image=16, return_img_only=True, num_sample_steps=None, clamp=True):

        # pmap_batch_size = jnp.array([batch_size // jax.device_count()])
        # shape = (batch_size, *self.sample_shape)
        # # rng, state: EMATrainState, shape, num_sample_steps = None, clamp = True)
        # samples = self._sample(rng, state, shape)
        # samples = jnp.reshape(samples, (-1, *self.sample_shape))
        # return samples

        b, h, w, c = lr_image.shape
        lr_image = jax.image.resize(lr_image, (b, h * self.sr_factor, w * self.sr_factor, c), method='bicubic')
        res = self._sample(key,state,lr_image.shape,x_self_cond=lr_image)
        if self.predict_residual:
            ret = res + lr_image
        else:
            ret = res

        if return_img_only:
            return [ret]
        else:
            return [ret, res, lr_image]

    """

    def sample(self, key, state, lr_image, return_img_only=True):

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

        if return_img_only:
            return ret
        else:
            return [ret, res, lr_image]

    """

    def __call__(self, key, state, params, img):

        b, h, w, c = img.shape
        lr_image = jax.image.resize(img, shape=(b, h // self.sr_factor, w // self.sr_factor, c), method='bilinear')
        fake_image = jax.image.resize(lr_image, shape=(b, h, w, c), method='bicubic')

        # real_image = residual + fake_image
        if self.predict_residual:
            x_start = img - fake_image
        else:
            x_start = img

        return self.p_loss(key, state, params, x_start, fake_image)
