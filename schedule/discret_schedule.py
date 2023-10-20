from collections import namedtuple
from functools import partial
from einops import einops
from flax.training.common_utils import shard, shard_prng_key
from modules.noise.noise import get_noise
from modules.gaussian.schedules import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
import jax
import jax.numpy as jnp

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(x):
    return x


class BasicSchedule:
    def __init__(
            self,
            train_timestep=1000,
            sample_timestep=100,
            objective='predict_noise',
            beta_schedule='linear',
            beta_schedule_configs=None,
            scale_shift=False,
            self_condition=False,
            noise_type='normal',
            scale_factor=1,
            mean=0,
            std=1,
            clip_x_min=-1,
            clip_x_max=1,
            clip_x_start=True,
            *args,
            **kwargs
    ):

        if beta_schedule_configs is None:
            beta_schedule_configs = {}
        self.clip_x_min = clip_x_min
        self.clip_x_max = clip_x_max
        self.clip_x_start = clip_x_start
        self.train_state = True
        self.noise_type = noise_type
        self.scale_factor = scale_factor
        self.model = None

        self.self_condition = self_condition
        self.mean = mean
        self.std = std

        assert objective in {'predict_noise', 'predict_x0', 'predict_v', 'predict_mx'}
        self.objective = objective

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise NotImplemented()

        betas = beta_schedule_fn(train_timestep, **beta_schedule_configs)

        alphas = 1 - betas
        if scale_shift:
            image_size = self.sample_shape[0]
            scale = 64 / image_size
            snr = alphas / (1 - alphas)
            alphas = 1 - 1 / (1 + scale ** 1 * snr)
            betas = 1 - alphas

        alphas_cumprod = jnp.cumprod(alphas)
        alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), constant_values=1)

        # print(alphas_cumprod)

        self.train_timestep = int(train_timestep)

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

        self.sample_timestep=sample_timestep

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

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

    def generate_noise(self, key, shape):
        return get_noise(self.noise_type, key, shape)

    def q_sample(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(self, x, t, model_output):
        clip_x_start = self.clip_x_start
        maybe_clip = partial(jnp.clip, a_min=self.clip_x_min, a_max=self.clip_x_max) if clip_x_start else identity

        if self.objective == 'predict_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            # if clip_x_start and rederive_pred_noise:
            #     pred_noise = self.predict_noise_from_start(x, t, x_start)

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
        else:
            raise NotImplemented()

        return ModelPrediction(pred_noise, x_start)
    """
    def step(self, model_output, timestep, samples):

        prev_timestep = timestep - self.num_timesteps // self.sampling_timesteps

        pred_noise, x_start = self.model_predictions(x=samples, t=timestep, model_output=model_output)

        prev_sample = jax.lax.cond(prev_timestep[0] < 0, lambda a, b, c: x_start,
                                   self.q_sample, x_start, prev_timestep, pred_noise)
        return prev_sample

    def _generate(self, key, params, self_condition=None, shape=None):

        b, *_ = shape
        key, key_image = jax.random.split(key, 2)
        samples = self.generate_noise(key_image, shape=shape)

        timesteps = (jnp.arange(0, self.sampling_timesteps) * self.train_train_timestep // self.sampling_timesteps)[
                    ::-1]

        def loop_body(step, in_args):
            samples = in_args

            batch_times = timesteps[step]
            timestep = jnp.broadcast_to(batch_times, samples.shape[0])

            model_output = self.unet.apply({'params': params}, samples, timestep)
            samples = self.step(model_output, timestep, samples)

            return samples

        # for i in range(self.sampling_timesteps):
        #     samples=loop_body(i, (samples))

        samples = jax.lax.fori_loop(0, self.sampling_timesteps, loop_body, init_val=(samples))

        return samples

    def generate(self, key, params, self_condition=None, shape=None, pmap=True):

        if pmap:
            b, *_ = shape
            b = b // jax.device_count()
            shape = (b, *_)

            samples = jax.pmap(self._generate, static_broadcasted_argnums=(3,))(shard_prng_key(key),
                                                                                shard(params),
                                                                                shard(self_condition),
                                                                                shape)
            samples = einops.rearrange(samples, 'n b h w c->(n b) h w c')
        else:
            samples = jax.jit(self._generate, static_argnums=(3,))(key, params, self_condition, shape)

        return samples
    """
