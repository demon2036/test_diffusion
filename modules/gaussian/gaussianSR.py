from modules.gaussian.gaussian import Gaussian, extract,ModelPrediction
import jax
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp


@jax.pmap
def model_predict(model, x, time):
    return model.apply_fn({"params": model.ema_params}, x, time)

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

    def p_sample_loop(self, key, params, shape, x_self_cond=None):
        key, normal_key = jax.random.split(key, 2)
        img = self.generate_nosie(normal_key, shape)

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            key, normal_key = jax.random.split(key, 2)
            img, x_start = self.p_sample(normal_key, params, img, t, x_self_cond)

        ret = img
        if self.predict_residual:
            ret += x_self_cond
        return ret

    def ddim_sample(self, key, shape, x_self_cond=None):
        b, *_ = shape
        key, key_image = jax.random.split(key, 2)
        img = self.generate_nosie(key_image, shape=shape)

        times = np.asarray(np.linspace(-1, 999, num=self.sampling_timesteps + 1), dtype=np.int32)
        times = list(reversed(times))

        for time, time_next in tqdm(zip(times[:-1], times[1:]), total=self.sampling_timesteps):
            batch_times = jnp.full((b,), time)
            pred_noise, x_start = self.model_predictions(None, img, batch_times, x_self_cond)

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
                img = self.q_sample(x_start, batch_times_next, noise)
        ret = img
        if self.predict_residual:
            ret += x_self_cond

        return ret

    def sample(self, key, params, lr_image):
        b, h, w, c = lr_image.shape
        lr_image = jax.image.resize(lr_image, (b, h * self.sr_factor, w * self.sr_factor, c), method='bicubic')
        noise_shape = lr_image.shape

        if self.num_timesteps > self.sampling_timesteps:
            return self.ddim_sample(key, noise_shape, lr_image)
        else:
            return self.p_sample_loop(key, params, noise_shape, lr_image)

    def p_loss(self, key, state, params, x_start, t):
        noise = self.generate_nosie(key, shape=x_start.shape)

        b, h, w, c = x_start.shape
        lr_image = jax.image.resize(x_start, shape=(b, h // self.sr_factor, w // self.sr_factor, c), method='bilinear')
        fake_image = jax.image.resize(lr_image, shape=(b, h, w, c), method='bicubic')

        if self.predict_residual:
            x_start = x_start - fake_image

        x = self.q_sample(x_start, t, noise)
        model_output = state.apply_fn({"params": params}, x, t, fake_image)

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
