import argparse
import os
import numpy as np
from modules.utils import read_yaml
from trainers.basic_trainer import Trainer


def cal_mean_std():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/preprocess/diff_ae/2D/ffhq-2D-z.yaml')
    args = parser.parse_args()

    config = read_yaml(args.config_path)
    config['train']['file_path']='dataraws'#config['train']['save_path']
    config['train']['data_type'] = 'np'
    trainer = Trainer(**config['train'], dataset_type='dataloader', drop_last=False)
    print(args)

    count = 0
    mean = 0
    std = 0
    max_value = 0
    min_value = 0
    for data in trainer.dl:
        mean += data.mean()
        std += data.std()

        # m=data.mean()
        # s = data.std()
        #
        # t = data - m
        # ss=data/s
        # print(t.mean(),ss.std())
        # mean += data.mean()
        # std += data.std()
        max_value = max(max_value, data.max())
        min_value = min(min_value, data.min())
        count += 1
    print(data.shape)
    mean /= count
    std /= count
    print(mean, std, max_value, min_value)

    # datas = []
    # for data in dl:
    #     datas.append(data)
    #
    # datas = np.array(datas)
    # print(datas.mean(), datas.std())


if __name__ == "__main__":
    cal_mean_std()

    pass
