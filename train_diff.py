import argparse
import jax.random
from modules.state_utils import create_obj_by_config, create_state_by_config
from modules.utils import  read_yaml
import os
from jax_smi import initialise_tracking
from trainers.diff_trainer import DiffTrainer

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/training/Diffusion/test.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)
    train_gaussian = create_obj_by_config(config['Gaussian'])
    train_state = create_state_by_config(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                         state_configs=config['State'])
    trainer = DiffTrainer(train_state, train_gaussian, **config['train'])
    trainer.load()
    # trainer.sample()
    trainer.sample_images(batch_size=512,save_path=trainer.save_path+f'/{trainer.finished_steps}')
    #trainer.train()