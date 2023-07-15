import os
import time
from multiprocessing.pool import Pool
import einops
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, CelebA
from torchvision.transforms import *
import os
from PIL import Image
import albumentations as A


def get_dataloader(batch_size=32, dataset='mnist', cache=True,image_size=64):
    if dataset == 'mnist':
        transform = Compose([
            ToTensor(),
        ])
        mnist = CIFAR10('./dataset', download=True, transform=transform)
        dataloader = DataLoader(mnist, batch_size=batch_size, num_workers=os.cpu_count(), persistent_workers=True,
                                drop_last=True)
        return dataloader
    else:
        mnist = MyDataSet(dataset, cache,image_size)
        dataloader = DataLoader(mnist, batch_size=batch_size, num_workers=os.cpu_count()//2, persistent_workers=True,
                                drop_last=True)
        return dataloader


class MyDataSet(Dataset):
    def __init__(self, path, cache=True,image_size=64):
        self.image_size=image_size
        self.path = path
        self.cache = cache
        self.data = []
        self.count = 0
        self.img_names = os.listdir(self.path)
        if self.cache:
            for img_name in tqdm.tqdm(self.img_names):
                self.data.append(self._preprocess(self.path + '/' + img_name))

    def _preprocess(self, image_path):
        img = Image.open(image_path)
        img = np.array(img)/255.0
        img = 2*img-1
        img = A.resize(img, self.image_size, self.image_size)
        return img

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.cache:
            img = self.data[idx]
        else:
            img = self._preprocess(self.path + '/' + self.img_names[idx])

        return img


if __name__ == '__main__':
    dl = get_dataloader(32, '/home/john/datasets/celeba-128/celeba-128',cache=True)

    start = time.time()
    for _ in range(100):
        for data in dl:
            print(type(data))
            # print(data.shape)
    end = time.time()
    print(end - start)
