import argparse
import os
import numpy as np

from data.dataset import get_dataloader
from modules.utils import read_yaml



def cal_mean_std():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='../configs/stat/config.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    dataloader_configs = config['dataloader_configs']
    dl = get_dataloader(**dataloader_configs)  # file_path

    count = 0
    mean = 0
    std = 0

    for data in dl:
        mean += data.mean()
        std += data.std()
        count += 1
    print(data.shape)
    mean /= count
    std /= count
    print(mean, std)

    # datas = []
    # for data in dl:
    #     datas.append(data)
    #
    # datas = np.array(datas)
    # print(datas.mean(), datas.std())








if __name__ == "__main__":
    cal_mean_std()


    pass




