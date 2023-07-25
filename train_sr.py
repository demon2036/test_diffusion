import argparse
from tqdm import tqdm
import jax.random

from dataset import generator
from unet import *
from schedules import *
from utils import *
import os
import time
from functools import partial
from flax.training import dynamic_scale as dynamic_scale_lib, train_state, orbax_utils
import optax
from flax.training.common_utils import shard, shard_prng_key
from collections import namedtuple
from jax_smi import initialise_tracking
from loss import l1_loss, l2_loss
from gaussian import Gaussian

initialise_tracking()

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class TrainState(train_state.TrainState):
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    ema_params: Any = None


@jax.pmap
def model_predict(model: TrainState, x, time):
    return model.apply_fn({"params": model.ema_params}, x, time)


def create_state(rng, model_cls, input_shape, optimizer, train_state, print_model=True, optimizer_kwargs=None,
                 model_kwargs=None, ):
    model = model_cls(**model_kwargs)
    if print_model:
        print(model.tabulate(rng, jnp.empty(input_shape), jnp.empty((input_shape[0],)), jnp.empty(input_shape), depth=2,
                             console_kwargs={'width': 200}))
    variables = model.init(rng, jnp.empty(input_shape), jnp.empty((input_shape[0],)), jnp.empty(input_shape))
    dynamic_scale = None
    if optimizer == 'AdamW':
        optimizer = optax.adamw
    elif optimizer == "Lion":
        optimizer = optax.lion
    else:
        assert "some thing is wrong"

    tx = optax.chain(
        optax.clip_by_global_norm(1),
        optimizer(**optimizer_kwargs)
    )
    return train_state.create(apply_fn=model.apply, params=variables['params'], tx=tx, dynamic_scale=dynamic_scale,
                              ema_params=variables['params'])


@partial(jax.pmap, static_broadcasted_argnums=(3), axis_name='batch')
def train_step(state: TrainState, batch, train_key, cls):
    def loss_fn(params):
        loss = cls(train_key, state, params, batch)
        return loss

    dynamic_scale = state.dynamic_scale
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, )  # axis_name=pmap_axis
        dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
        # grad_fn = dynamic_scale.value_and_grad(cls.p_loss, argnums=1)  # axis_name=pmap_axis
        # dynamic_scale, is_fin, loss, grads = grad_fn(state.params,state,key,batch)
    else:
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
        grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"loss": loss}

    return new_state, metric


class test(Gaussian):
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
                # noise = self.generate_nosie(key_noise, shape=shape)
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


def sample_save_image(key, diffusion: test, steps, state: TrainState, batch, save_path):
    os.makedirs(save_path,exist_ok=True)
    b, h, w, c = batch.shape
    lr_image = jax.image.resize(batch, (b, h // diffusion.sr_factor, w // diffusion.sr_factor, c), method='bilinear')
    os.makedirs(save_path, exist_ok=True)
    diffusion.set_state(state)
    sample = diffusion.sample(key, state.ema_params, lr_image)
    all_image = jnp.concatenate([sample, batch], axis=0)
    sample = all_image / 2 + 0.5
    diffusion.state = None
    sample = einops.rearrange(sample, 'b h w c->b c h w')
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'{save_path}/{steps}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./sr4.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    unet_config = config['Unet']
    gaussian_config = config['Gaussian']

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs, optimizer, optimizer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)

    diffusion = test(**gaussian_config, image_size=dataloader_configs['image_size'])

    state = create_state(rng=key, model_cls=Unet, input_shape=input_shape, optimizer=optimizer,
                         optimizer_kwargs=optimizer_configs,
                         train_state=TrainState, model_kwargs=unet_config)

    model_ckpt = {'model': state, 'steps': 0}
    model_save_path = trainer_configs['model_path']
    checkpoint_manager = create_checkpoint_manager(model_save_path, max_to_keep=10)
    if len(os.listdir(model_save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = model_ckpt['model']

    state = flax.jax_utils.replicate(model_ckpt['model'])
    dl = generator(**dataloader_configs)  # file_path
    finished_steps = model_ckpt['steps']

    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, trainer_configs['total_steps']):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            batch = shard(batch)
            state, metrics = train_step(state, batch, train_step_key, diffusion)
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            pbar.set_postfix(metrics)
            pbar.update(1)

            if steps > 100:
                state = update_ema(state, 0.9999)

            if steps % trainer_configs['sample_steps'] == 0:
                batch = einops.rearrange(batch, 'n b h w c -> (n b ) h w c')
                sample_save_image(key, diffusion, steps, state, batch, save_path=trainer_configs['save_path'])
                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}  # 'steps': steps
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)
                # del unreplicate_state, sample, model_ckpt
