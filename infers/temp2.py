import jax
import argparse

from modules.infer_utils import jax_img_save
from modules.state_utils import create_state, create_obj_by_config, create_state_by_config, EMATrainState
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt, get_obj_from_str
import os

import flax
from trainers.diff_sr_trainer import DiffSRTrainer
from trainers.diff_trainer import DiffTrainer
from trainers.ldm_trainer import LdmTrainer

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def get_diff_trainer(args):
    diff_config = read_yaml(args.config_path_diff)
    train_gaussian = create_obj_by_config(diff_config['Gaussian'])
    train_state = create_state_by_config(rng=jax.random.PRNGKey(seed=diff_config['train']['seed']),
                                         state_configs=diff_config['State'])
    return DiffTrainer(train_state, train_gaussian, **diff_config['train'])


def get_diff_sr_trainer(args):
    diff_config = read_yaml(args.config_path_diff_sr)
    train_gaussian = create_obj_by_config(diff_config['Gaussian'])
    train_state = create_state_by_config(rng=jax.random.PRNGKey(seed=diff_config['train']['seed']),
                                         state_configs=diff_config['State'])
    return DiffSRTrainer(train_state, train_gaussian, **diff_config['train'])


def go():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cpd', '--config_path_diff', default='./configs/training/ldm_2d/test.yaml')
    parser.add_argument('-cpsr', '--config_path_diff_sr', default='./configs/training/ldm_2d/test.yaml')
    args = parser.parse_args()
    print(args)
    trainer = get_diff_trainer(args)
    trainer.load(only_ema=True)
    sr_trainer = get_diff_sr_trainer(args)
    sr_trainer.load()
    # sample = trainer.sample(batch_size=8,return_sample=True,save_sample=False)
    sample = next(sr_trainer.dl)
    b, h, w, c = sample.shape
    sample = jax.image.resize(sample, (b, h // 8, w // 8, c), method='bilinear')
    sample_sr = sr_trainer.sample(batch=sample, return_sample=True)

    try:
        jax_img_save(sample_sr, save_path='./result', steps='sr')
        jax_img_save(sample, save_path='./result', steps='origin')
    except Exception as e:
        print(e)


if __name__ == "__main__":
    go()
