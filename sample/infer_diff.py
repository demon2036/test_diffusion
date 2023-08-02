import argparse
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import jax.random
from data.dataset import generator, get_dataloader, torch_to_jax
from modules.state_utils import create_state
from modules.utils import EMATrainState, create_checkpoint_manager, load_ckpt, read_yaml, update_ema, \
    sample_save_image_diffusion, get_obj_from_str
import flax
import os
from functools import partial
from flax.training import orbax_utils
from flax.training.common_utils import shard, shard_prng_key
from jax_smi import initialise_tracking
from modules.gaussian.gaussian import Gaussian
import jax.numpy as jnp

from tools.resize_dataset import save_image

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


@partial(jax.pmap, static_broadcasted_argnums=(3), axis_name='batch')
def train_step(state, batch, train_key, cls):
    def loss_fn(params):
        loss = cls(train_key, state, params, batch)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"loss": loss}
    return new_state, metric


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='../configs/Sample/Diffusion/test_diff.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, model_optimizer, unet_config = config['Unet'].values()
    model_cls = get_obj_from_str(model_cls_str)
    gaussian_config = config['Gaussian']

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)
    input_shapes = (input_shape, input_shape[0])

    c = Gaussian(**gaussian_config, image_size=dataloader_configs['image_size'])

    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=unet_config)

    model_ckpt = {'model': state, 'steps': 0}
    model_save_path = trainer_configs['model_path']

    checkpoint_manager = create_checkpoint_manager(model_save_path, max_to_keep=10)
    if len(os.listdir(model_save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    count = 0
    with ThreadPoolExecutor() as pool:
        with tqdm(total=30000//dataloader_configs['batch_size']) as pbar2:
            key, train_step_key = jax.random.split(key, num=2)
            sample=c.sample(key,state,None,batch_size=128)
            for x in sample:
                pool.submit(save_image, x, count, './data')
                count += 1

            pbar2.update(1)


if __name__ == "__main__":
    train()
