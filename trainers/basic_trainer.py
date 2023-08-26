import jax

from data.dataset import generator, get_dataloader
from modules.utils import create_checkpoint_manager


class Trainer:
    def __init__(self,
                 image_size=None,
                 batch_size=1,
                 file_path='/root/data/latent2D-128-8',
                 cache=False,
                 data_type='img',
                 repeat=1,
                 drop_last=True,
                 shuffle=True,
                 dataset_type='generator',
                 seed=43,
                 total_steps=3000000,
                 sample_steps=50000,
                 save_path='result/Diffusion',
                 model_path='check_points/Diffusion',
                 ckpt_max_to_keep=5
                 ):
        self.data_type = data_type

        assert dataset_type in ['generator', 'dataloader']
        if dataset_type == 'generator':
            self.dl = generator(batch_size, file_path, image_size, cache, data_type, repeat, drop_last, shuffle)
        else:
            self.dl = get_dataloader(batch_size, file_path, image_size, cache, data_type, repeat, drop_last, shuffle)
        self.rng = jax.random.PRNGKey(seed)
        self.total_steps = total_steps
        self.sample_steps = sample_steps
        self.save_path = save_path
        self.model_path = model_path
        self.checkpoint_manager = create_checkpoint_manager(model_path, max_to_keep=ckpt_max_to_keep)
        self.finished_steps = 1
