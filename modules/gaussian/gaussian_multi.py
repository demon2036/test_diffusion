from collections import namedtuple
from functools import partial
import numpy as np
from einops import einops
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm

from modules.gaussian.gaussian import Gaussian
from modules.noise.noise import normal_noise, truncate_noise, pyramid_nosie, resize_noise, offset_noise
from modules.gaussian.schedules import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from modules.loss.loss import l1_loss, l2_loss, charbonnier_loss
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


def identity(x):
    return x


class GaussianMulti(Gaussian):
    def __init__(
            self,
            sample_shape,
            loss='l2',
            timesteps=1000,
            sampling_timesteps=1000,
            objective='predict_noise',
            beta_schedule='linear',
            beta_schedule_configs=None,
            ddim_sampling_eta=0.,
            min_snr_loss_weight=False,
            scale_shift=False,
            self_condition=False,
            noise_type='normal',
            scale_factor=1,
            p_loss=False,
            mean=0,
            std=1,
            clip_x_start=True

    ):

        self.sample_shape = sample_shape
        if beta_schedule_configs is None:
            beta_schedule_configs = {}
        self.clip_x_start = clip_x_start
        self.train_state = True
        self.noise_type = noise_type
        self.scale_factor = scale_factor
        self.model = None

        self.self_condition = self_condition
        self.mean = mean
        self.std = std

        if mean != 0 and std != 1:
            print(f'mean:{mean} std:{std}')

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule

        betas = beta_schedule_fn(timesteps, **beta_schedule_configs)

        alphas = 1 - betas
        if scale_shift:
            image_size = self.sample_shape[0]
            scale = 64 / image_size
            snr = alphas / (1 - alphas)
            alphas = 1 - 1 / (1 + (scale) ** 1 * snr)
            betas = 1 - alphas

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
        self.snr = snr

        if loss == 'l2':
            self.loss = l2_loss
        elif loss == 'l1':
            self.loss = l1_loss
        elif loss == 'charbonnier':
            self.loss = charbonnier_loss

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr = jnp.clip(maybe_clipped_snr, a_max=5)

        self.loss_weight_noise = maybe_clipped_snr / snr
        self.loss_weight_x0 = maybe_clipped_snr / snr  # maybe_clipped_snr
        self.loss_weight_v = maybe_clipped_snr / (snr + 1)
        self.loss_weight_mx = snr ** 0.5

        self.pmap_q_sample = jax.pmap(self.q_sample)
        self.pmap_model_predictions = jax.pmap(self.model_predictions, static_broadcasted_argnums=(0,))
        self.pmap_p_sample = jax.pmap(self.p_sample)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

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

    def model_predictions(self, objective=None, x=None, t=None, x_self_cond=None, state=None, rederive_pred_noise=False,
                          *args, **kwargs):
        if self.train_state:
            model_output = model_predict(state, x, t, x_self_cond)
        else:
            model_output = model_predict_ema(state, x, t, x_self_cond)

        clip_x_start = self.clip_x_start
        maybe_clip = partial(jnp.clip, a_min=-1., a_max=1.) if clip_x_start else identity

        if objective == 'predict_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif objective == 'predict_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif objective == 'predict_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif objective == 'predict_mx':
            x_start = -model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def ddim_sample(self, key, states, self_condition=None, shape=None, gaussian_configs=None):
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

        count = 0

        for time, time_next in tqdm(zip(times[:-1], times[1:]), total=self.sampling_timesteps):
            batch_times = jnp.full((b,), time)

            if has_condition:
                pass
            elif self.self_condition:
                x_self_cond = x_start
            else:
                x_self_cond = None

            batch_times = shard(batch_times)

            cls = gaussian_configs[-int(count + 1)]
            state = states[-int(count + 1)]
            # print(cls.objective, cls.time_min, cls.time_max)

            if time < cls.time_min:
                count += 1

            pred_noise, x_start = self.pmap_model_predictions(cls.objective, x=img, t=batch_times,
                                                              x_self_cond=x_self_cond,
                                                              state=state, )

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

        # img = einops.rearrange(img, 'n b h w c->(n b ) h w c', n=img.shape[0])
        print(img.shape)

        img = jnp.reshape(img, (-1, *self.sample_shape))

        return img

    def sample(self, key, states, self_condition=None, batch_size=64, gaussian_configs=None):
        if self_condition is not None:
            batch_size = self_condition.shape[0]

        if self.num_timesteps > self.sampling_timesteps:
            samples = self.ddim_sample(key, states, self_condition, (batch_size, *self.sample_shape), gaussian_configs)
        else:
            samples = self.p_sample_loop(key, states, self_condition, (batch_size, *self.sample_shape))

        # output image will be denormalized by mean(default as 0) and std(default as 1) because input image was
        # normalized if mean=0 and std=1 img=denormalize(image)
        samples = samples / self.scale_factor
        samples = self.denormalize(samples)

        return samples

    def p_loss(self, key, state, params, x_start, t, gaussian_conf):
        key, cond_key = jax.random.split(key, 2)
        noise = self.generate_nosie(key, shape=x_start.shape)

        x_start = x_start * self.scale_factor
        # noise sample
        x = self.q_sample(x_start, t, noise)

        print(gaussian_conf.objective)

        def estimate(_):
            return jax.lax.stop_gradient(
                self.model_predictions(gaussian_conf.objective, x, t, state=state, x_self_cond=jnp.zeros_like(x), )
            ).pred_x_start

        zeros = jnp.zeros_like(x)
        x_self_cond = None
        if self.self_condition:
            x_self_cond = jax.lax.cond(jax.random.uniform(cond_key, shape=(1,))[0] < 0.5, estimate, lambda _: zeros,
                                       None)

        model_output = state.apply_fn({"params": params}, x, t, x_self_cond)

        objective = gaussian_conf.objective
        print(objective)

        if objective == 'predict_noise':
            target = noise
            loss_weight = self.loss_weight_noise
        elif objective == 'predict_x0':
            target = x_start
            loss_weight = self.loss_weight_x0
        elif objective == 'predict_v':
            v = self.predict_v(x_start, t, noise)
            target = v
            loss_weight = self.loss_weight_v
        elif objective == 'predict_mx':
            target = -x_start
            loss_weight = self.loss_weight_mx
        else:
            raise NotImplemented()

        p_loss = self.loss(target, model_output)

        p_loss = (p_loss * extract(loss_weight, t, p_loss.shape)).mean()
        return p_loss

    def __call__(self, key, state, params, img, gaussian_conf):
        # input image will be normalized by mean(default as 0) and std(default as 1)
        # if mean=0 and std=1 img=normalize(image)
        img = self.normalize(img)

        key_times, key_noise = jax.random.split(key, 2)
        b, *_ = img.shape

        time_min = jnp.clip(gaussian_conf.time_min - 50, a_min=0)

        time_max = jnp.clip(gaussian_conf.time_max + 50, a_max=self.num_timesteps)

        print(f'time_min:{time_min} time_max:{time_max}')

        t = jax.random.randint(key_times, (b,), minval=time_min, maxval=time_max)

        return self.p_loss(key_noise, state, params, img, t, gaussian_conf)
