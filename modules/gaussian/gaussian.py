from collections import namedtuple
from functools import partial
import numpy as np
from einops import einops
from flax.training.common_utils import shard
from tqdm import tqdm

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


def model_predict_ema(model, x, time, x_self_cond=None):
    return model.apply_fn({"params": model.ema_params}, x, time, x_self_cond)


def model_predict(model, x, time, x_self_cond=None):
    return model.apply_fn({"params": model.params}, x, time, x_self_cond)


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Gaussian:
    def __init__(
            self,
            loss='l2',
            image_size=32,
            timesteps=1000,
            sampling_timesteps=1000,
            objective='predict_noise',
            beta_schedule='linear',
            ddim_sampling_eta=0.,
            min_snr_loss_weight=False,
            scale_shift=False,
            self_condition=False

    ):
        self.scale = 1
        self.state = None
        self.model = None
        self.image_size = image_size
        self.self_condition = self_condition
        assert objective in {'predict_noise', 'predict_x0', 'predict_v'}
        self.objective = objective

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule

        betas = beta_schedule_fn(timesteps)

        alphas = 1 - betas

        if scale_shift:
            scale = 64 / image_size
            snr = alphas / (1 - alphas)
            alphas = 1 - 1 / (1 + (scale) ** 1 * snr)

        alphas_cumprod = jnp.cumprod(alphas)
        alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), constant_values=1)

        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = sampling_timesteps

        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sample = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1 / alphas_cumprod)
        # self.sqrt_recip_one_minus_alphas_cumprod = jnp.sqrt(1 / (1 - alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1 / alphas_cumprod - 1)

        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.posterior_variance = posterior_variance

        self.posterior_log_variance_clipped = jnp.log(posterior_variance.clip(min=1e-20))
        self.posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1 - alphas_cumprod)

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if loss == 'l2':
            self.loss = l2_loss
        elif loss == 'l1':
            self.loss = l1_loss

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr = jnp.clip(maybe_clipped_snr, a_max=5)

        if objective == 'predict_noise':
            self.loss_weight = maybe_clipped_snr / snr
        elif objective == 'predict_x0':
            self.loss_weight = maybe_clipped_snr
        elif objective == 'predict_v':
            self.loss_weight = maybe_clipped_snr / (snr + 1)

        self.pmap_q_sample = jax.pmap(self.q_sample)
        self.pmap_model_predictions = jax.pmap(self.model_predictions)
        self.pmap_p_sample = jax.pmap(self.p_sample)

    def set_state(self, state):
        self.state = state

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        pass

    def model_predictions(self, x, t=None, x_self_cond=None, state=None, rederive_pred_noise=False, *args, **kwargs):
        # model_output = model_predict(state, x, t, x_self_cond)
        model_output = model_predict_ema(state, x, t, x_self_cond)

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

    def p_mean_variance(self, x, t, x_self_cond=None, *args, **kwargs):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start

        # x_start = jnp.clip(x_start, 0, 1)
        # if clip_denoised:
        #     x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def generate_nosie(self, key, shape):
        return jax.random.normal(key, shape) * self.scale

    def p_sample(self, key, x, t, x_self_cond=None):
        b, c, h, w = x.shape
        batch_times = jnp.full((b,), t)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batch_times, x_self_cond)
        noise = self.generate_nosie(key, x.shape)
        pred_image = model_mean + jnp.exp(0.5 * model_log_variance) * noise

        return pred_image, x_start

    def p_sample_loop(self, key, shape):
        key, normal_key = jax.random.split(key, 2)
        img = self.generate_nosie(normal_key, shape)

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            key, normal_key = jax.random.split(key, 2)
            x_self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(normal_key, img, t, x_self_cond)

        ret = img

        return ret

    def ddim_sample(self, key, state, self_condition=None, shape=None):
        print(shape)
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

        x_start = None
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

        img = einops.rearrange(img, 'n b h w c->(n b ) h w c', n=img.shape[0])

        return img

    def sample(self, key, state, self_condition=None):
        batch_size = self_condition.shape[0]

        return self.ddim_sample(key, state, self_condition, (batch_size, self.image_size, self.image_size, 3))

        # if self.num_timesteps > self.sampling_timesteps:
        #     return self.ddim_sample(key, (batch_size, self.image_size, self.image_size, 3))
        # else:
        #     return self.p_sample_loop(key, (batch_size, self.image_size, self.image_size, 3))

    def q_sample(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_loss(self, key, state, params, x_start, t):
        key, cond_key = jax.random.split(key, 2)
        noise = self.generate_nosie(key, shape=x_start.shape)

        # noise sample
        x = self.q_sample(x_start, t, noise)

        def estimate(_):
            return jax.lax.stop_gradient(self.model_predictions(None, x, t, state=state)).pred_x_start

        zeros = jnp.zeros_like(x)
        x_self_cond = None
        if self.self_condition:
            x_self_cond = jax.lax.cond(jax.random.uniform(cond_key, shape=(1,))[0] < 0.5, estimate, lambda _: zeros,
                                       None)

        model_output = state.apply_fn({"params": params}, x, t, x_self_cond)

        if self.objective == 'predict_noise':
            target = noise
        elif self.objective == 'predict_x0':
            target = x_start
        elif self.objective == 'predict_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            target = None

        p_loss = self.loss(target, model_output)

        p_loss = (p_loss * extract(self.loss_weight, t, p_loss.shape)).mean()
        return p_loss

    def __call__(self, key, state, params, img):
        key_times, key_noise = jax.random.split(key, 2)
        b, h, w, c = img.shape
        t = jax.random.randint(key_times, (b,), minval=0, maxval=self.num_timesteps)

        return self.p_loss(key_noise, state, params, img, t)
