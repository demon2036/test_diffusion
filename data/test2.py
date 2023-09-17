import argparse

import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='/home/john/data/FFHQ')
    args = parser.parse_args()
    print(args)
    dataset = wds.WebDataset(args.config_path).shuffle(1000).decode('rgb').to_tuple('jpg')
    dl = DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True)
    for sample in tqdm(dl):
        print(sample[0].shape)
        data = sample[0] / 2 + 0.5
