from math import sqrt

from flax.training.common_utils import shard, shard_prng_key
from flax.training.train_state import TrainState
import flax.training.train_state
import torch
import jax
import jax.numpy as jnp
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from modules.loss.loss import l2_loss, l1_loss, charbonnier_loss
from modules.noise.noise import normal_noise, resize_noise, truncate_noise, offset_noise, pyramid_nosie
from modules.state_utils import EMATrainState


def block_noise( g_noise, block_size=1, ):
    if block_size == 1:
        return g_noise

    blk_noise = jnp.zeros(g_noise.shape, )
    for px in range(block_size):
        for py in range(block_size):
            blk_noise += jnp.roll(g_noise, shift=(px, py), axis=(1, 2))

    blk_noise = blk_noise / block_size  # to maintain the same std on each pixel
    return blk_noise

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# tensor helpers

# def log(t, eps=1e-20):
#     return torch.log(t.clamp(min=eps))

def log(t, eps=1e-20):
    return jnp.log(jnp.clip(t, a_min=eps))


# normalization functions


# main class

class ElucidatedDiffusion:
    def __init__(
            self,
            sample_shape,
            loss,
            self_condition=False,
            block_size=1,
            noise_type='normal',
            num_sample_steps=32,  # number of sampling steps
            sigma_min=0.002,  # min noise level
            sigma_max=80,  # max noise level
            sigma_data=0.5,  # standard deviation of data distribution
            rho=7,  # controls the sampling schedule
            P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
    ):

        self.self_condition = self_condition
        self.noise_type = noise_type
        self.block_size=block_size

        # image dimensions
        self.sample_shape = sample_shape
        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        if loss == 'l2':
            self.loss = l2_loss
        elif loss == 'l1':
            self.loss = l1_loss
        elif loss == 'charbonnier':
            self.loss = charbonnier_loss

        self.train_state = True
        self.pmap_preconditioned_network_forward = jax.pmap(self.preconditioned_network_forward)
        self.pmap_sample = jax.pmap(self._sample, )

    # derived preconditioning params - Table 1

    def generate_noise(self, key, shape):
        if self.noise_type == 'normal':
            gen_noise = normal_noise(key, shape)
        elif self.noise_type == 'truncate':
            gen_noise = truncate_noise(key, shape)
        elif self.noise_type == 'resize':
            gen_noise = resize_noise(key, shape)
        elif self.noise_type == 'offset':
            gen_noise = offset_noise(key, shape)
        elif self.noise_type == 'pyramid':
            gen_noise = pyramid_nosie(key, shape)
        else:
            raise NotImplemented()
        return block_noise(gen_noise,self.block_size)

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma, self_cond=None, clamp=False,
                                       state: EMATrainState = None, params=None):
        batch = noised_images.shape[0]

        if params is None:
            if self.train_state:
                params = state.params
            else:
                params = state.ema_params

        # if isinstance(sigma, float):
        #     sigma = jnp.full((batch,), sigma)
        #     # sigma = torch.full((batch,), sigma, device=device)
        # else:
        #     print(type(sigma))

        sigma = jnp.full((batch,), sigma)
        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')
        # padded_sigma = sigma

        # if params is None:

        net_out = state.apply_fn({'params': params},
                                 self.c_in(padded_sigma) * noised_images,
                                 self.c_noise(sigma),
                                 self_cond
                                 )

        out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out

        if clamp:
            out = jnp.clip(out, -1., 1.)
            # out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = jnp.arange(num_sample_steps, dtype=jnp.float32)
        # steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = jnp.pad(sigmas, (0, 1), constant_values=0.)
        # sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas

    def _sample(self, rng, state, shape, num_sample_steps=None, x_self_cond=None):

        if self.train_state:
            params = state.params
        else:
            params = state.ema_params
        # params = flax.jax_utils.replicate(params)

        rng_noise, rng_sample = jax.random.split(rng)

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        # shape = (8, *self.sample_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        # images = init_sigma * torch.randn(shape, device=self.device)
        images = init_sigma * jax.random.normal(rng_noise, shape, )

        # for self conditioning

        if x_self_cond is not None:
            x_self_cond = shard(x_self_cond)
            has_condition = True
        else:
            has_condition = False

        x_start = None

        # gradually denoise
        # for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step'):
        for sigma, sigma_next in tqdm(sigmas_and_gammas, desc='sampling time step'):
            rng_noise, rng_sample = jax.random.split(rng_sample)

            # sigma, sigma_next = map(lambda t: t.item(), (sigma, sigma_next))
            gamma = min(self.S_churn / num_sample_steps, sqrt(2) - 1) if self.S_tmin <= sigma <= self.S_tmax else 0

            # eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling
            eps = self.S_noise * jax.random.normal(rng_sample, shape, )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            # sigma_hat = flax.jax_utils.replicate(sigma_hat)
            # images_hat = shard(images_hat)
            # sigma_next = flax.jax_utils.replicate(sigma_next)

            # self_cond = x_start if self.self_condition else None
            if has_condition:
                pass
            elif self.self_condition:
                x_self_cond = x_start
            else:
                x_self_cond = None

            # images_hat = shard(images_hat)
            # sigma_hat = shard(sigma_hat)
            # model_output = self.pmap_preconditioned_network_forward(images_hat, sigma_hat,  # self_cond, clamp=clamp,
            #                                                         state=state, params=params)
            model_output = self.pmap_preconditioned_network_forward(shard(images_hat),
                                                                    flax.jax_utils.replicate(sigma_hat),  x_self_cond,
                                                                    # self_cond, clamp=clamp,
                                                                    state=state, params=params)

            model_output = model_output.reshape(-1, *self.sample_shape)

            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                # model_output_next = self.pmap_preconditioned_network_forward(images_next, sigma_next,  # self_cond,
                #                                                              # clamp=flax.jax_utils.replicate(clamp),
                #                                                              state=state, params=params)

                model_output_next = self.pmap_preconditioned_network_forward(shard(images_next),
                                                                             flax.jax_utils.replicate(sigma_next),
                                                                             x_self_cond,
                                                                             # clamp=flax.jax_utils.replicate(clamp),
                                                                             state=state, params=params)

                model_output_next = model_output_next.reshape(-1, *self.sample_shape)

                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output

            x_start = x_start.reshape(-1, *self.sample_shape)

        # images = images.clamp(-1., 1.)

        images = jnp.clip(images, -1, 1)

        return images

    def sample(self, rng, state: EMATrainState, batch_size=16, num_sample_steps=None, ):

        # pmap_batch_size = jnp.array([batch_size // jax.device_count()])
        shape = (batch_size, *self.sample_shape)
        # rng, state: EMATrainState, shape, num_sample_steps = None, clamp = True)
        samples = self._sample(rng, state, shape)
        samples = jnp.reshape(samples, (-1, *self.sample_shape))
        return samples

    # def sample_using_dpmpp(self, batch_size=16, num_sample_steps=None):
    #     """
    #     thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
    #     https://arxiv.org/abs/2211.01095
    #     """
    #
    #     device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)
    #
    #     sigmas = self.sample_schedule(num_sample_steps)
    #
    #     shape = (batch_size, self.channels, self.image_size, self.image_size)
    #     images = sigmas[0] * torch.randn(shape, device=device)
    #
    #     sigma_fn = lambda t: t.neg().exp()
    #     t_fn = lambda sigma: sigma.log().neg()
    #
    #     old_denoised = None
    #     for i in tqdm(range(len(sigmas) - 1)):
    #         denoised = self.preconditioned_network_forward(images, sigmas[i].item())
    #         t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
    #         h = t_next - t
    #
    #         if not exists(old_denoised) or sigmas[i + 1] == 0:
    #             denoised_d = denoised
    #         else:
    #             h_last = t - t_fn(sigmas[i - 1])
    #             r = h_last / h
    #             gamma = - 1 / (2 * r)
    #             denoised_d = (1 - gamma) * denoised + gamma * old_denoised
    #
    #         images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
    #         old_denoised = denoised
    #
    #     images = images.clamp(-1., 1.)
    #     return unnormalize_to_zero_to_one(images)

    # training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, key, batch_size):
        # return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()
        noise = self.generate_noise(key, (batch_size,))
        return jnp.exp(self.P_mean + self.P_std * noise)

    def p_loss(self, key, state, params, images, x_self_cond=None):

        key_noise, key_sigmas, key_noise = jax.random.split(key, 3)

        batch_size = images.shape[0]

        assert images.shape[1:] == tuple(self.sample_shape)

        sigmas = self.noise_distribution(key_sigmas, batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = self.generate_noise(key_noise, images.shape)

        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        # todo : implement self condition
        #
        # if self.self_condition and random() < 0.5:
        #     # from hinton's group's bit diffusion paper
        #     with torch.no_grad():
        #         self_cond = self.preconditioned_network_forward(noised_images, sigmas)
        #         self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_images, sigmas, x_self_cond, state=state, params=params)

        losses = self.loss(denoised, images)
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()

    def __call__(self, key, state, params, images):
        return self.p_loss(key, state, params, images)

    def train(self):
        self.train_state = True

    def eval(self):
        self.train_state = False
