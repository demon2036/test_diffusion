import einops
import flax
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
from flax.training.common_utils import shard
from torchvision.utils import save_image

from data.dataset import generator, get_dataloader
from modules.models.autoencoder import AutoEncoder
from functools import partial
import jax
import jax.numpy as jnp
from modules.loss.loss import l1_loss, l2_loss, hinge_d_loss
import argparse

from modules.models.diffEncoder import DiffEncoder
from modules.state_utils import create_state
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt, update_ema, sample_save_image_autoencoder, \
    get_obj_from_str, EMATrainState
import os

from tqdm import tqdm

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def save_latent(x, count, save_path):
    np.save(file=f'{save_path}/{count}.npy', arr=x)


@jax.pmap
def diff_encode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=DiffEncoder.encode)


def decode(ae_state: EMATrainState, sample_latent, first_stage_gaussian):
    first_stage_gaussian.eval()
    sample = first_stage_gaussian.sample(key, ae_state, sample_latent)
    return sample


def get_auto_encoder_diff(config):
    ae_cls_str, model_optimizer, model_configs = config['AutoEncoder'].values()
    gaussian, gaussian_configs = get_obj_from_str(config['Gaussian']['target']), config['Gaussian']['params']

    print(config['Gaussian']['target'])
    first_stage_gaussian = gaussian(**gaussian_configs)
    ae_cls = get_obj_from_str(ae_cls_str)

    key = jax.random.PRNGKey(seed=43)
    input_shape = (1, 128, 128, 3)
    input_shapes = (input_shape, input_shape[0], input_shape)
    state = create_state(rng=key, model_cls=ae_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=model_configs)

    model_ckpt = {'model': state, 'steps': 0}
    save_path = './model/DiffAE'
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=1)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    return state, first_stage_gaussian


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/preprocess/diff_ae/2D/ffhq256-2D-128-latent8.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    key = jax.random.PRNGKey(seed=43)
    state, first_stage_gaussian = get_auto_encoder_diff(config['FirstStage'])

    dataloader_configs, trainer_configs = train_config.values()

    dl = get_dataloader(**dataloader_configs, drop_last=False)  # file_path
    save_path = '/home/john/data/latent2D-128-8'
    os.makedirs(save_path, exist_ok=True)
    count = 0

    with ThreadPoolExecutor() as pool:
        for data in tqdm(dl):
            x = data
            x = x.numpy()
            x = jnp.asarray(x)

            x = shard(x)
            sample_latent = diff_encode(state, x)
            sample_latent = jnp.reshape(sample_latent, (-1, *sample_latent.shape[2:]))
            latent = np.array(sample_latent, dtype='float32')
            for x in latent:
                pool.submit(save_latent, x, count, save_path)
                count += 1

            # print(sample_latent.shape)
            # y = decode(state, sample_latent, first_stage_gaussian)
            # y = jnp.concatenate([y, jnp.reshape(x, (-1, *x.shape[2:]))])
            # sample = y / 2 + 0.5
            # sample = einops.rearrange(sample, '( b) h w c->( b ) c h w', )
            # sample = np.array(sample)
            # sample = torch.Tensor(sample)
            # save_image(sample, f'test.png')
            # break
