import argparse
from tqdm import tqdm
import jax.random
from modules.models.unet import *
from modules.gaussian.schedules import *
from modules.utils import *
import os
from functools import partial
from flax.training import dynamic_scale as dynamic_scale_lib, train_state
import optax
from flax.training.common_utils import shard
from collections import namedtuple

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

def l2_loss(predictions, target):
    return optax.l2_loss(predictions=predictions, targets=target)


def l1_loss(predictions, target):
    return jnp.abs(target - predictions)


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class TrainState(train_state.TrainState):
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    ema_params: Any = None


@jax.pmap
def model_predict(model: TrainState, x):
    return model.apply_fn({"params": model.ema_params}, x)


def create_state(rng, model_cls, input_shape, learning_rate, train_state, print_model=True, model_kwargs=None, *args, ):
    platform = jax.local_devices()[0].platform

    if platform == "gpu":
        dynamic_scale = dynamic_scale_lib.DynamicScale()
        dynamic_scale = None
    else:
        dynamic_scale = None

    model = model_cls(*args, **model_kwargs)
    if print_model:
        print(model.tabulate(rng, jnp.empty(input_shape), depth=2, console_kwargs={'width': 120}))
    variables = model.init(rng, jnp.empty(input_shape))
    tx = optax.lion(learning_rate, weight_decay=1e-2)
    return train_state.create(apply_fn=model.apply, params=variables['params'], tx=tx, dynamic_scale=dynamic_scale,
                              ema_params=variables['params'])


@partial(jax.pmap, static_broadcasted_argnums=(3), axis_name='batch')
def train_step(state: TrainState, batch, train_key, cls):
    def loss_fn(params):
        loss = cls(train_key, params, batch)
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


class test:
    def __init__(
            self,
            model=Unet,
            model_kwargs={},
            loss='l2',
            image_size=32,
            timesteps=1000,
            sampling_timesteps=1000,
            objective='predict_noise',
            beta_schedule='linear',
            ddim_sampling_eta=0.,
    ):
        self.state = None
        self.model = model(**model_kwargs)
        self.image_size = image_size
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
        self.sqrt_recip_one_minus_alphas_cumprod = jnp.sqrt(1 / (1 - alphas_cumprod))
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

    def set_state(self, state):
        self.state = state

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            # extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recip_one_minus_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * x0
        )

    def predict_v(self, x_start, t, noise):
        pass

    def predict_start_from_v(self, x_t, t, v):
        pass

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        pass

    def model_predictions(self, params, x, t, *args, **kwargs):
        x = shard(x)
        model_out = model_predict(self.state, x)
        model_out = einops.rearrange(model_out, 'n b h w c->(n b) h w c')
        x = einops.rearrange(x, 'n b h w c->(n b) h w c')
        clip_x_start = True
        maybe_clip = partial(jnp.clip, a_min=-1., a_max=1.) if clip_x_start else identity

        if self.objective == 'predict_noise':
            pred_noise = model_out
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            # pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'predict_x0':
            x_start = model_out
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, params, x, t, x_self_cond=None, *args, **kwargs):
        preds = self.model_predictions(params, x, t)
        x_start = preds.pred_x_start

        # x_start = jnp.clip(x_start, 0, 1)
        # if clip_denoised:
        #     x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def generate_nosie(self, key, shape):
        return jax.random.normal(key, shape)

    def p_sample(self, key, params, x, t):
        b, c, h, w = x.shape
        batch_times = jnp.full((b,), t)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(params, x, batch_times)
        noise = self.generate_nosie(key, x.shape)
        pred_image = model_mean + jnp.exp(0.5 * model_log_variance) * noise

        return pred_image, x_start

    def p_sample_loop(self, key, params, shape):
        key, normal_key = jax.random.split(key, 2)
        img = self.generate_nosie(normal_key, shape)

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            key, normal_key = jax.random.split(key, 2)
            img, x_start = self.p_sample(normal_key, params, img, t)

        ret = img

        return ret

    def ddim_sample(self, key, shape):
        b, *_ = shape
        key, key_image = jax.random.split(key, 2)
        img = self.generate_nosie(key_image, shape=shape)

        times = np.asarray(np.linspace(-1, 999, num=self.sampling_timesteps + 1), dtype=np.int32)
        times = list(reversed(times))

        for time, time_next in tqdm(zip(times[:-1], times[1:]), total=self.sampling_timesteps):
            batch_times = jnp.full((b,), time)
            pred_noise, x_start = self.model_predictions(None, img, batch_times)

            if time_next < 0:
                img = x_start
            else:
                key, key_noise = jax.random.split(key, 2)
                noise = self.generate_nosie(key_noise, shape=shape)
                # noise = pred_noise
                batch_times_next = jnp.full((b,), time_next)
                img = self.q_sample(x_start, batch_times_next, noise)

        return img

    def sample(self, key, params, batch_size=16):
        if self.num_timesteps > self.sampling_timesteps:
            return self.ddim_sample(key, (batch_size, self.image_size, self.image_size, 3))
        else:
            return self.p_sample_loop(key, params, (batch_size, self.image_size, self.image_size, 3))

    def q_sample(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_loss(self, key, params, x_start, t):
        noise = self.generate_nosie(key, shape=x_start.shape)

        # noise sample
        x = self.q_sample(x_start, t, noise)
        model_output = self.model.apply({"params": params}, x)

        if self.objective == 'predict_noise':
            target = noise
        elif self.objective == 'predict_x0':
            target = x_start
        else:
            target = None

        p_loss = self.loss(target, model_output).mean()
        return p_loss

    def __call__(self, key, params, img):
        key_times, key_noise = jax.random.split(key, 2)
        b, h, w, c = img.shape
        t = jax.random.randint(key_times, (b,), minval=0, maxval=self.num_timesteps)

        return self.p_loss(key_noise, params, img, t)


def generator(batch_size=32, file_path='/home/john/datasets/celeba-128/celeba-128', image_size=64):
    # d = get_dataloader(batch_size.)
    d = get_dataloader(batch_size, file_path, cache=True, image_size=image_size)
    while True:
        for data in d:
            # x, y = data
            x = data
            x = x.numpy()
            x = jnp.asarray(x)
            # x = einops.rearrange(x, 'b c h w-> b h w c')
            # b, h, w, c = x.shape
            # x = jax.image.resize(x, (b, 32, 32, c), method='bicubic')
            yield x


@partial(jax.pmap, static_broadcasted_argnums=(1,))
def update_ema(state: TrainState, ema_decay=0.999):
    new_ema_params = jax.tree_map(lambda ema, normal: ema * ema_decay + (1 - ema_decay) * normal, state.ema_params,
                                  state.params)
    state = state.replace(ema_params=new_ema_params)
    return state


def sample_save_image(key, c, steps, state: TrainState):
    c.set_state(state)
    sample = c.sample(key, state.ema_params, batch_size=64)
    sample = sample / 2 + 0.5
    c.state = None
    sample = einops.rearrange(sample, 'b h w c->b c h w')
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'./result/{steps}.png')


if __name__ == "__main__":
    # if os.path.exists('./nohup.out'):
    #    os.remove('./nohup.out')

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./ae4.yaml')
    # parser.add_argument('-ct', '--continues',action=True ,)
    args = parser.parse_args()
    print(args)

    config = read_yaml(args.config_path)

    train_config = config['train']
    unet_config = config['Unet']
    gaussian_config = config['Gaussian']

    key = jax.random.PRNGKey(seed=43)
    image_size, seed, batch_size, data_path = train_config.values()

    print(image_size, seed, batch_size, data_path)

    input_shape = (16, image_size, image_size, 3)

    c = test(model_kwargs=unet_config, **gaussian_config, image_size=image_size)

    state = create_state(rng=key, model_cls=Unet, input_shape=input_shape, learning_rate=1e-5,
                         train_state=TrainState, model_kwargs=unet_config)

    model_ckpt = {'model': state, 'steps': 0}
    save_path = '../check_points'
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=50)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = model_ckpt['model']

    state = flax.jax_utils.replicate(model_ckpt['model'])
    time_steps = [20, 25, 35, 50, 75, 100, 200, 250, 500, 1000]
    for time in time_steps:
        c.sampling_timesteps = time
        sample_save_image(key, c, time, state)
