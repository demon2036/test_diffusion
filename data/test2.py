import argparse

import einops
import numpy as np
import torchvision.utils
from webdataset import WebLoader
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


def test(x):
    # x=np.array([0])
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='/home/john/data/FFHQ.tar')
    args = parser.parse_args()
    print(args)
    # /root/fused_bucket/temp/FFHQ256_split-{000000..000069}.tar
    dataset = wds.WebDataset('/home/john/data/test/_test-{000000..000069}.tar').shuffle(1000).decode(
        'rgb').to_tuple('jpg', ).map_tuple(test)
    dl = DataLoader(dataset, batch_size=64, num_workers=8)
    for epoch in range(2):
        for i, sample in enumerate(tqdm(dl, )):
            # print(sample[0].shape)
            # data = sample[0] / 2 + 0.5
            if i == 0:
                x = sample[0]
                x = einops.rearrange(x, 'b h w c->b c h w')
                torchvision.utils.save_image(x, f'test_{epoch}.png')
