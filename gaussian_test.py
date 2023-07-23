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
from gaussian import Gaussian
initialise_tracking()


os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'





class TrainState(train_state.TrainState):
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    ema_params: Any = None





def create_state(rng, model_cls, input_shape, learning_rate, optimizer, train_state, print_model=True,
                 model_kwargs=None, *args, ):
    platform = jax.local_devices()[0].platform

    if platform == "gpu":
        dynamic_scale = dynamic_scale_lib.DynamicScale()
        dynamic_scale = None
    else:
        dynamic_scale = None

    model = model_cls(*args, **model_kwargs)
    if print_model:
        print(model.tabulate(rng, jnp.empty(input_shape), jnp.empty((input_shape[0],)), depth=2,
                             console_kwargs={'width': 200}))
    variables = model.init(rng, jnp.empty(input_shape), jnp.empty((input_shape[0],)))

    if optimizer == 'AdamW':
        optimizer = optax.adamw
    elif optimizer == "Lion":
        optimizer = optax.lion
    else:
        assert "soem thing is wrong"

    tx = optax.chain(
        optax.clip_by_global_norm(1),
        optimizer(learning_rate, weight_decay=1e-2)
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

    os.makedirs('./result', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./test.yaml')
    # parser.add_argument('-ct', '--continues',action=True ,)
    args = parser.parse_args()
    print(args)

    config = read_yaml(args.config_path)

    train_config = config['train']
    unet_config = config['Unet']
    gaussian_config = config['Gaussian']

    key = jax.random.PRNGKey(seed=43)

    image_size, seed, batch_size, data_path, \
        learning_rate, optimizer, sample_steps = train_config.values()

    print(train_config.values())

    input_shape = (1, image_size, image_size, 3)

    c = Gaussian(**gaussian_config, image_size=image_size)

    state = create_state(rng=key, model_cls=Unet, input_shape=input_shape, learning_rate=learning_rate,
                         optimizer=optimizer,
                         train_state=TrainState, model_kwargs=unet_config)

    model_ckpt = {'model': state, 'steps': 0}
    save_path = './check_points/Unet'
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=50)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = model_ckpt['model']

    state = flax.jax_utils.replicate(model_ckpt['model'])
    dl = generator(batch_size=batch_size, image_size=image_size, file_path=data_path)  # file_path
    finished_steps = model_ckpt['steps']

    with tqdm(total=1000000) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, 1000000):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            batch = shard(batch)
            state, metrics = train_step(state, batch, train_step_key, c)
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            pbar.set_postfix(metrics)
            pbar.update(1)

            if steps > 100:
                state = update_ema(state, 0.9999)

            if steps % sample_steps == 0:
                try:
                    sample_save_image(key, c, steps, state)
                except Exception as e:
                    print(e)

                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}  # 'steps': steps
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)
                # del unreplicate_state, sample, model_ckpt

    end = time.time()

    """
    c.set_state(state)
    sample = c.sample(key, state.params, batch_size=64)
    c.state = None
    sample = einops.rearrange(sample, 'b h w c->b c h w')
    sample = np.array(sample)
    sample = torch.Tensor(sample)
    save_image(sample, f'./result/test.png')
    
    """
