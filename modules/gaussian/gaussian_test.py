from collections import namedtuple
from functools import partial
import numpy as np
from einops import einops
from flax.jax_utils import replicate
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm

from modules.gaussian.gaussian import Gaussian, identity
from modules.models.diffEncoder import DiffEncoder
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


def model_predict_ema(model, x, time, x_self_cond=None, method=None):
    print(f'method:{method}')
    return model.apply_fn({"params": model.ema_params}, x, time, x_self_cond, method=method)


def model_predict(model, x, time, x_self_cond=None):
    return model.apply_fn({"params": model.params}, x, time, x_self_cond)


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianTest(Gaussian):
    def __init__(
            self,
            *args,
            **kwargs

    ):
        super().__init__(*args, **kwargs)
        self.apply_fn = DiffEncoder.decode

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
