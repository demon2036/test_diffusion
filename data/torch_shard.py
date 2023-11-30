import io
import os
import cv2
import einops
import numpy
import numpy as np
import torch
import torchvision.transforms
import webdataset
import webdataset as wds
from timm.data import RandAugment, create_transform
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
import jax
import jax.numpy as jnp

MEAN_RGB = [0, 0, 0]
STDDEV_RGB = [255, 255, 255]

mean = jnp.array(MEAN_RGB, dtype=np.float32).reshape(1, 1, 3)
std = jnp.array(STDDEV_RGB, dtype=np.float32).reshape(1, 1, 3)


def test(x):
    x = x['jpg']
    x = np.array(x)
    x = A.HorizontalFlip()(image=x)['image']
    x = A.SmallestMaxSize(256, cv2.INTER_AREA)(image=x)['image']
    x = A.RandomCrop(256, 256)(image=x)['image']

    return x
    # return {'image': x, 'label': cls}


def normalize(images):
    # images = images.float()
    # print(images.dtype)
    images -= mean
    images /= std
    return images


def prepare_torch_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = numpy.asarray(x)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    xs = jax.tree_util.tree_map(_prepare, xs)

    xs['image'] = jax.pmap(normalize)(xs['image'])

    return xs


def create_input_pipeline_torch(file_path=None,num_workers=jax.device_count()*2,batch_size=64,*args, **kwargs):
    # urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00073..00073}.tar '
    # urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/laion_tar/imagenet_train_shards-{00000..00221}.tar '
    urls=file_path
    # urls = 'pipe:gsutil cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00000..00073}.tar '

    #urls = 'pipe: cat /home/john/data/imagenet_train_shards/imagenet_train_shards-{00073..00073}.tar'

    dataset = wds.WebDataset(
        urls=urls,
        shardshuffle=False,handler=wds.ignore_and_continue).mcached().decode('pil').map(
        test)  # .batched(1024,collation_fn=default_collate).map(temp)

    dataloader = wds.WebLoader(dataset, num_workers=16, prefetch_factor=2, batch_size=batch_size, drop_last=True,
                           )

    while True:
        for xs in dataloader:
            # del xs['__key__']
            xs = xs / 255.0
            xs = (xs*2)-1
            xs = np.asarray(xs)
            # xs = jax.pmap(normalize)(xs)

            yield xs


if __name__ == "__main__":
    dl = create_input_pipeline_torch()

    data = next(dl)

    images = np.asarray(data)
    images = torch.Tensor(images)

    from torchvision.utils import save_image

    images = einops.rearrange(images, 'b h w c->( b ) c h w  ')
    print(images.shape)

    save_image(images, 'test.png')
