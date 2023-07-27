import os
import time
import einops
import flax.linen
import numpy as np
import cv2
import torch
import torchvision.utils
import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, CelebA
from torchvision.transforms import *
import os
from PIL import Image
import albumentations as A
import jax.numpy as jnp


def get_dataloader(batch_size=32, dataset='/home/john/data/s', cache=True, image_size=64,repeat=1):
    data = MyDataSet(dataset, cache, image_size,repeat=repeat)

    dataloader = DataLoader(data, batch_size=batch_size,
                            num_workers=min(os.cpu_count(), len(data.data) // batch_size // 2)
                            , persistent_workers=True, pin_memory=True, shuffle=True,
                            drop_last=True)
    return dataloader


class MyDataSet(Dataset):
    def __init__(self, path, cache=True, image_size=64,repeat=1):
        self.repeat=repeat
        self.image_size = image_size
        self.path = path
        self.cache = cache
        self.data = []
        self.count = 0
        self.img_names = os.listdir(self.path)

        if self.cache:
            for _ in range(self.repeat):
                for img_name in tqdm.tqdm(self.img_names):
                    self.data.append(self._preprocess(self.path + '/' + img_name))
        else:
            self.data = self.img_names

        self.real_length = len(self.data)

    def _preprocess(self, image_path):
        img = Image.open(image_path)
        img = np.array(img) / 255.0
        img = 2 * img - 1
        img = A.smallest_max_size(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = A.center_crop(img,self.image_size,self.image_size)
        #img = A.random_crop(img, self.image_size, self.image_size, 0, 0)
        return img

    def __len__(self):
        return len(self.img_names)*self.repeat

    def __getitem__(self, idx):
        if self.cache:
            img = self.data[idx]
        else:
            img = self._preprocess(self.path + '/' + self.img_names[idx%self.real_length])

        return img

def generator(batch_size=32, file_path='/home/john/datasets/celeba-128/celeba-128', image_size=64, cache=False):
    d = get_dataloader(batch_size, file_path, cache=cache, image_size=image_size)
    while True:
        for data in d:
            x = data
            x = x.numpy()
            x = jnp.asarray(x)
            yield x


if __name__ == '__main__':
    start = time.time()
    dl = get_dataloader(64, '/home/john/data/s', cache=False, image_size=256,repeat=2)
    end = time.time()
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    # for _ in range(100):
    #     for data in dl:
    #         print(type(data))
    #         # print(data.shape)

    for data in dl:
        print(data.shape)
        data = data / 2 + 0.5
        data=data.numpy()
        data=jnp.asarray(data)
        data=np.asarray(data)
        data=torch.Tensor(data)


        data = einops.rearrange(data, '(n b) h w c->(b n) c h w',n=2)
        torchvision.utils.save_image(data, './test2.png')
        break

    print(end - start)
