import einops
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from data.dataset import generator, get_dataloader
from modules.models.autoencoder import AutoEncoder
from functools import partial
import jax
import jax.numpy as jnp
from modules.loss.loss import l1_loss, l2_loss, hinge_d_loss
import argparse

from modules.state_utils import create_state
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt, update_ema, sample_save_image_autoencoder, \
    get_obj_from_str, EMATrainState
import os

from tqdm import tqdm

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def save_latent(x, count, save_path):
    np.save(file=f'{save_path}/{count}.npy', arr=x)


@partial(jax.jit)
def encode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=AutoEncoder.encode)


@partial(jax.jit)
def decode(state: EMATrainState, x):
    return state.apply_fn({'params': state.ema_params}, x, method=AutoEncoder.decode)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/AutoEncoder/test_gan.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_config = config['train']
    model_cls_str, model_optimizer, model_configs = config['AutoEncoder'].values()
    print(model_cls_str)
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
    state = model_ckpt['model']
    # state = flax.jax_utils.replicate(model_ckpt['model'])

    dl = get_dataloader(**dataloader_configs, drop_last=False)  # file_path
    ae = AutoEncoder(**model_configs)
    save_path = '/home/john/data/latent'
    os.makedirs(save_path, exist_ok=True)
    count = 0
    with ThreadPoolExecutor() as pool:
        for data in tqdm(dl):
            x = data
            x = x.numpy()
            x = jnp.asarray(x)
            latent = encode(state, x)
            latent = np.array(latent, dtype='float32')
            for x in latent:
                pool.submit(save_latent, x, count, save_path)
                count += 1

        # y = decode(state, latent)
        # sample = y / 2 + 0.5
        # sample = einops.rearrange(sample, '( b) h w c->(b ) c h w', )
        # sample = np.array(sample)
        # sample = torch.Tensor(sample)
        # save_image(sample, f'test.png')
        # break
