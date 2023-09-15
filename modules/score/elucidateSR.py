import einops
import jax
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm
import jax.numpy as jnp
from modules.score.elucidate import ElucidatedDiffusion
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

    def sample(self, key, state: EMATrainState, lr_image=16, return_img_only=True, num_sample_steps=None, clamp=True):

        b, h, w, c = lr_image.shape
        lr_image = jax.image.resize(lr_image, (b, h * self.sr_factor, w * self.sr_factor, c), method='bicubic')
        res = self._sample(key, state, lr_image.shape, x_self_cond=lr_image)
        if self.predict_residual:
            ret = res + lr_image
        else:
            ret = res

        if return_img_only:
            return ret
        else:
            return [ret, res, lr_image]

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
