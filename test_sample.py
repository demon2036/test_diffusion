import argparse
import time

import einops
import jax.random
from flax.training.common_utils import shard, shard_prng_key

from data.dataset import jax_to_torch
from modules.state_utils import create_obj_by_config, create_state_by_config
from modules.utils import read_yaml
import os
from jax_smi import initialise_tracking
from trainers.diff_trainer import DiffTrainer
from schedule.ddim_schedule import DDIMSchedule
import numpy as np

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
import imageio


def save_gif(images, save_name='output.gif', rows=4, columns=None):
    n, b, h, w, c = images.shape

    if rows is not None and columns is not None:
        assert b == rows * columns
    elif rows is not None:
        columns = b // rows
    elif columns is not None:
        rows = b // columns

    images = einops.rearrange(images, 'n (b1 b2) h w c-> n (b1 h) (b2 w ) c', b1=rows, b2=columns)

    frames = [np.asarray(_ * 255, dtype=np.uint8) for _ in images / 2 + 0.5]

    # Save as a GIF
    imageio.mimsave(save_name, frames, duration=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/training/Diffusion/test.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_gaussian = create_obj_by_config(config['Gaussian'])
    train_state, model = create_state_by_config(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                                state_configs=config['State'],
                                                return_model=True)

    trainer = DiffTrainer(train_state, train_gaussian, **config['train'])
    key = jax.random.PRNGKey(config['train']['seed'])
    data = next(trainer.dl)
    data = einops.rearrange(data, 'n b h w c->(n b) h w c')
    print(data.shape)

    b, h, w, c = data.shape
    t = jax.numpy.full((b,), 100)
    noise = train_gaussian.generate_nosie(key, data.shape)
    noise_data = train_gaussian.q_sample(data, t, noise)

    trainer.load()

    start = time.time()

    shape = (64, 64, 3)
    sampler = DDIMSchedule(sample_shape=shape, unet=model, return_intermediate=False, test_data=noise_data)

    sample_shape = (8,) + shape

    samples = sampler.generate(key, trainer.state.params, None, sample_shape, pmap=True)

    end = time.time()
    print(end - start)
    # save_gif(inter)

    samples = samples / 2 + 0.5
    print(samples.shape)

    # inter = einops.rearrange(inter, 'n (b1 b2) h w c-> n (b1 h) (b2 w ) c', b1=4)

    from torchvision.utils import save_image

    samples = einops.rearrange(samples, 'b h w c->b c h w ')
    samples = jax_to_torch(samples)

    save_image(samples, 'test2.png', nrow=10)
    """
    """
