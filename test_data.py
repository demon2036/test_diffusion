import io
import os

import numpy
import numpy as np
import torch
import webdataset
import webdataset as wds
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
import jax
import jax.numpy as jnp


# .batched(1024)


def collect_fn(x):
    # print(x)
    return x


class MyDataSet(Dataset):
    def __init__(self, cache_data):
        self.cache_data = cache_data

        self.transform = A.Resize(256, 256)

    def __len__(self):
        return len(self.cache_data)

    def _preprocess(self, x):
        x = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
        x = np.array(x)
        x = self.transform(image=x)['image']
        return x

    def __getitem__(self, index):
        x = self._preprocess(self.cache_data[index])
        return x


def test(x):
    cls = int(x['cls'].decode('utf-8'))
    x = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
    x = np.array(x)
    x = A.Resize(224, 224)(image=x)['image']

    return {'img': x, 'cls': torch.nn.functional.one_hot(torch.Tensor(np.array(cls).reshape(-1)).to(torch.int64), 1000)}


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()
    xs = {'images': xs['img'], 'labels': xs['cls']}

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        # x = {'img': x['img'], 'cls': x['cls']}
        x = numpy.asarray(x)
        # x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


if __name__ == "__main__":
    urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00000..000075}.tar '
    #urls = 'pipe: cat /home/john/data/imagenet_train_shards/imagenet_train_shards-{00000..00073}.tar'
    dataset = wds.WebDataset(
        urls=urls,
        shardshuffle=False).map(test)  # .mcached().map(test)

    dl = DataLoader(dataset, num_workers=24, prefetch_factor=4, batch_size=1024,
                    # collate_fn=collect_fn,
                    persistent_workers=True)

    dl = map(prepare_tf_data,dl, )

    for data in tqdm(dl):
        pass

        # print(data)
        # print(data)

    for _ in range(100):
        for data in tqdm(dl):
            pass

    """
    temp = []

    for data in tqdm(dl):
        temp.extend(data)

    print(len(temp))

    dataset = MyDataSet(temp)

    dl = DataLoader(dataset, num_workers=jax.device_count() * 2, prefetch_factor=16, batch_size=1024)

    for data in tqdm(dl):
        # print(data.shape)
        pass
    """
