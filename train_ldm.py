import jax
import argparse
from modules.state_utils import create_state, create_obj_by_config, create_state_by_config, EMATrainState
from modules.utils import read_yaml, create_checkpoint_manager, load_ckpt,get_obj_from_str
import os
import flax

from trainers.ldm_trainer import LdmTrainer

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def get_auto_encoder_diff(config):
    ae_cls_str, model_optimizer, model_configs = config['AutoEncoder'].values()
    first_stage_gaussian = create_obj_by_config(config['Gaussian'])

    ae_cls = get_obj_from_str(ae_cls_str)

    key = jax.random.PRNGKey(seed=43)
    input_shape = (1, 256, 256, 3)
    input_shapes = (input_shape, input_shape[0], input_shape)
    state = create_state(rng=key, model_cls=ae_cls, input_shapes=input_shapes,
                         optimizer_dict=model_optimizer,
                         train_state=EMATrainState, model_kwargs=model_configs)

    model_ckpt = {'model': state, 'steps': 0}
    save_path = './check_points/DiffAE'
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=1)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    model_ckpt['model'] = model_ckpt['model'].replace(params=None)

    state = flax.jax_utils.replicate(model_ckpt['model'])
    return state, first_stage_gaussian


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/training/ldm_2d/test.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)

    train_gaussian = create_obj_by_config(config['Gaussian'])
    first_stage_config = config['FirstStage']
    ae_state, first_stage_gaussian = get_auto_encoder_diff(first_stage_config)
    train_state = create_state_by_config(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                         state_configs=config['State'])
    trainer = LdmTrainer(train_state, train_gaussian, ae_state, first_stage_gaussian, **config['train'])
    trainer.load()
    # trainer.sample()
    trainer.train()
