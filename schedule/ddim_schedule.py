import time
from collections import namedtuple
from functools import partial

import flax.jax_utils
from einops import einops
from flax.training.common_utils import shard, shard_prng_key
from modules.noise.noise import get_noise
import jax
import jax.numpy as jnp
from schedule.discret_schedule import BasicSchedule
from diffusers.schedulers import FlaxDDIMScheduler
from diffusers import FlaxStableDiffusionPipeline
from jax.experimental import host_callback

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

from diffusers import FlaxStableDiffusionPipeline, DDIMScheduler


class DDIMSchedule(BasicSchedule):
    def __init__(
            self,
            unet,
            dynamic_thresholding_ratio=0.995,
            return_intermediate=True,
            *args,
            **kwargs

    ):
        super().__init__(*args, **kwargs)
        self.intermediate_values = []
        self.unet = unet
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.return_intermediate = return_intermediate

    def _threshold_sample(self, sample):
        b, h, w, c = sample.shape
        abs_data = jnp.abs(sample)
        abs_data = einops.rearrange(abs_data, 'b h w c->b (h w c)')
        q = jnp.quantile(abs_data, self.dynamic_thresholding_ratio, axis=1)
        q = jnp.expand_dims(q, axis=(1, 2, 3))
        sample = jnp.clip(sample, -q, q) / q
        return sample

    def step(self, model_output, timestep, samples):

        prev_timestep = timestep - self.train_timestep // self.sample_timestep

        pred_noise, x_start = self.model_predictions(x=samples, t=timestep, model_output=model_output)

        x_start = self._threshold_sample(x_start)

        prev_sample = jax.lax.cond(prev_timestep[0] < 0, lambda a, b, c: x_start,
                                   self.q_sample, x_start, prev_timestep, pred_noise)

        return prev_sample

    def _generate(self, key, params, self_condition=None, shape=None):

        b, *_ = shape
        key, key_image = jax.random.split(key, 2)
        samples = self.generate_noise(key_image, shape=shape)

        timesteps = (jnp.arange(0, self.sample_timestep) * self.train_timestep // self.sample_timestep)[::-1]
        print(len(timesteps))

        intermediate_values = []

        # def collect_callback(intermediate, transform):
        #     print(intermediate.shape)
        #     self.intermediate_values.append(intermediate)

        def collect_callback(intermediate):
            print(intermediate.shape)
            self.intermediate_values.append(intermediate)
            # print(self.intermediate_values)
            return intermediate

        def loop_body(step, in_args):
            samples = in_args

            batch_times = timesteps[step]
            timestep = jnp.broadcast_to(batch_times, samples.shape[0])

            model_output = self.unet.apply({'params': params}, samples, timestep)
            samples = self.step(model_output, timestep, samples)

            if self.return_intermediate:
                result_shape_dtype = jax.ShapeDtypeStruct(
                    shape=samples.shape,
                    dtype=samples.dtype
                )

                # jax.pure_callback(collect_callback,result_shape_dtype,samples)
                # host_callback.id_tap(collect_callback, samples)
                jax.experimental.io_callback(collect_callback, result_shape_dtype, samples)

            return samples

        samples = jax.lax.fori_loop(0, self.sample_timestep, loop_body, init_val=(samples))

        print(samples.shape)
        #
        # intermediate_values = jnp.concatenate(intermediate_values)
        #
        # print(intermediate_values.shape)
        # print(f'hers is intermediate_values:{intermediate_values.shape}')

        return samples

    def generate(self, key, params, self_condition=None, shape=None, pmap=True):

        if pmap:
            b, *_ = shape
            b = b // jax.device_count()
            shape = (b, *_)

            samples = jax.pmap(self._generate, static_broadcasted_argnums=(3,))(shard_prng_key(key),
                                                                                flax.jax_utils.replicate(params),
                                                                                shard(self_condition),
                                                                                shape)
            samples = einops.rearrange(samples, 'n b h w c->(n b) h w c')
        else:
            samples = jax.jit(self._generate, static_argnums=(3,))(key, params, self_condition, shape)
            # samples = self._generate(key, params, self_condition, shape)

        # samples.block_until_ready()
        host_callback.barrier_wait()
        intermediate_values = jnp.concatenate(self.intermediate_values)
        print(intermediate_values.shape)

        return samples
