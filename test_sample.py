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

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

if __name__ == "__main__":
    """
   

    
    import jax.numpy as jnp

    from jax.experimental import host_callback

    # test=jnp.
    x = jnp.zeros((10,) + (1,))

    temp = []


    def test(value, transforms):
        print(value, transforms)
        temp.append(value)


    # def loop_body(step, in_args):
    #     array, _ = in_args
    #     array = array.at[step].set(step)
    #
    #     host_callback.id_tap(test, _)
    #
    #     return array, _
    #
    #
    # in_args = jax.lax.fori_loop(0, 10, loop_body, init_val=(x, jnp.array([1])))
    #

    def loop_body(step, in_args):
        data = in_args
        host_callback.id_tap(test, data)
        data = data + 1
        return data


    in_args = jax.lax.fori_loop(0, 10, loop_body, init_val=(jnp.array([1])))
    print(in_args)
    host_callback.barrier_wait()
    print(temp)
    """

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
    trainer.load()

    start=time.time()

    shape = (64, 64, 3)
    sampler = DDIMSchedule(sample_shape=shape, unet=model,return_intermediate=True)

    key = jax.random.PRNGKey(config['train']['seed'])
    sample_shape = (8,) + shape

    samples = sampler.generate(key, trainer.state.params, None, sample_shape, pmap=True)
    samples = samples / 2 + 0.5
    print(samples.shape)

    end=time.time()
    print(end-start)


    from torchvision.utils import save_image

    samples = einops.rearrange(samples, 'b h w c->b c h w ')
    samples = jax_to_torch(samples)
    save_image(samples, 'test2.png')
    


    # import matplotlib.pyplot as plt
    #
    # plt.imshow(samples[0])
    # plt.show()

    """
    # trainer.sample()
    # trainer.sample_images(batch_size=512,save_path=trainer.save_path+f'/{trainer.finished_steps}')
    trainer.train()
    """
