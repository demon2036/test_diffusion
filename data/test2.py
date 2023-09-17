import argparse
from webdataset import WebLoader
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='/home/john/data/FFHQ.tar')
    args = parser.parse_args()
    print(args)
    dataset = wds.WebDataset('//root/fused_bucket/temp/FFHQ256_split-{000000..000069}.tar').shuffle(1000).decode('rgb').to_tuple('jpg')
    dl = DataLoader(dataset, batch_size=64, num_workers=8)
    for sample in tqdm(dl):
        print(sample[0].shape)
        data = sample[0] / 2 + 0.5
