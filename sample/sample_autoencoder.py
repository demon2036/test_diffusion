import einops
from flax.training import orbax_utils
from flax.training.common_utils import shard_prng_key, shard
from data.dataset import generator, get_dataloader, torch_to_jax
from modules.models.autoencoder import AutoEncoder
from functools import partial
import jax
import jax.numpy as jnp
from modules.loss.loss import l1_loss, l2_loss, hinge_d_loss
import optax
import argparse
from tools.resize_dataset import save_image

from modules.save_utils import save_image_from_jax
from modules.state_utils import create_state
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt, update_ema, sample_save_image_autoencoder, \
    get_obj_from_str, EMATrainState
import os
import flax
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


@jax.pmap
def encode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=AutoEncoder.encode)


@jax.pmap
def decode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=AutoEncoder.decode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='../configs/AutoEncoder/test_gan.yaml')
    parser.add_argument('-sp', '--save_path', default='./data/AutoEncoder')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, model_optimizer, model_configs = config['AutoEncoder'].values()
    model_cls = get_obj_from_str(model_cls_str)

    disc_cls_str, disc_optimizer, disc_configs = config['Discriminator'].values()
    disc_cls = get_obj_from_str(disc_cls_str)

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)
    input_shapes = (input_shape,)

    state = create_state(rng=key, model_cls=model_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=model_configs)

    discriminator_state = create_state(rng=key, model_cls=disc_cls, input_shapes=input_shapes,
                                       optimizer_dict=disc_optimizer,
                                       train_state=EMATrainState, model_kwargs=disc_configs)

    model_ckpt = {'model': state, 'discriminator': discriminator_state, 'steps': 0}
    save_path = trainer_configs['model_path']
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=1)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    discriminator_state = flax.jax_utils.replicate(model_ckpt['discriminator'])

    dl = generator(**dataloader_configs)  # file_path
    count = 0
    with ThreadPoolExecutor() as pool:
        for data in tqdm(dl):
            key, train_step_key = jax.random.split(key, num=2)
            x = torch_to_jax(data)
            x = shard(x)
            latent = encode(state, x)
            y = decode(state, latent)
            y = einops.rearrange(y, 'n b h w  c->(n b) c h w')
            y = y / 2 + 0.5
            for x in y:
                pool.submit(save_image, x, count, args.save_path)
                count += 1

            # save_image_from_jax(y,'./test')
            # break
