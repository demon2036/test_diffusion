import flax
import jax
from flax.training.common_utils import shard
import tensorflow_datasets as tfds
from data.dataset import generator, get_dataloader, MyDataSet, SRDataSet
from data.input_pipeline import create_split
from modules.utils import create_checkpoint_manager


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x  # x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


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
                 dataset=MyDataSet,
                 seed=43,
                 total_steps=3000000,
                 sample_steps=50000,
                 save_path='result/Diffusion',
                 model_path='check_points/Diffusion',
                 ckpt_max_to_keep=5
                 ):
        self.data_type = data_type

        assert dataset_type in ['generator', 'dataloader', 'tfds']
        if dataset_type == 'generator':
            self.dl = generator(batch_size, file_path, image_size, cache, data_type, repeat, drop_last, shuffle,
                                dataset)
        elif dataset_type == 'tfds':
            dataset_builder = tfds.builder('imagenet2012', try_gcs=True, data_dir=file_path)
            self.dl = create_split(dataset_builder, batch_size=batch_size, image_size=image_size, cache=True)
            self.dl=map(prepare_tf_data,self.dl)
        else:
            self.dl = get_dataloader(batch_size, file_path, image_size, cache, data_type, repeat, drop_last, shuffle,
                                     dataset)
        self.dl = map(shard, self.dl)
        self.dl = flax.jax_utils.prefetch_to_device(self.dl, 2)

        self.rng = jax.random.PRNGKey(seed)
        self.total_steps = total_steps
        self.sample_steps = sample_steps
        self.save_path = save_path
        self.model_path = model_path
        self.checkpoint_manager = create_checkpoint_manager(model_path, max_to_keep=ckpt_max_to_keep)
        self.finished_steps = 0
