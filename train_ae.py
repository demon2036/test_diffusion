import argparse
import os
import jax.random
from jax_smi import initialise_tracking
from modules.state_utils import create_obj_by_config, create_state_by_config
from modules.utils import read_yaml
from trainers.aegan_trainer import AutoEncoderTrainer

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/training/AE/test.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)

    train_state = create_state_by_config(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                         state_configs=config['State'])

    disc_state = create_state_by_config(rng=jax.random.PRNGKey(seed=config['train']['seed']),state_configs=config['Disc_State'])

    trainer = AutoEncoderTrainer(train_state, disc_state, **config['train'])
    # trainer.load()
    #trainer.sample()
    trainer.train()
