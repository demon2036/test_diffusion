import os

import webdataset
import webdataset as wds
from tqdm import tqdm
from torch.utils.data import DataLoader

data = wds.WebDataset(urls='pipe:gsutil cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00500..00600}.tar ',shardshuffle=False)#.batched(1024)


def collect_fn(x):
    # print(x)

    return x

dl = DataLoader(data, num_workers=48, prefetch_factor=4,batch_size=512,collate_fn=collect_fn)

temp = []
for x in tqdm(dl):
    temp.append(x)

print(len(temp))
