from ldm.autoencoder import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/AutoEncoder/test.yaml')
    # parser.add_argument('-ct', '--continues',action=True ,)

    args = parser.parse_args()
    config = read_yaml(args.config_path)

    train_config = config['train']
    autoencoder_config = config['AutoEncoder']

    key = jax.random.PRNGKey(seed=43)

    dataloader_configs, trainer_configs, optimizer, optimizer_configs = train_config.values()

    input_shape = (1, dataloader_configs['image_size'], dataloader_configs['image_size'], 3)

    state = create_state(rng=key, model_cls=AutoEncoder, input_shape=input_shape, optimizer=optimizer,
                         model_kwargs=autoencoder_config, optimizer_kwargs=optimizer_configs)

    model_ckpt = {'model': state, 'steps': 0}
    save_path = trainer_configs['model_path']
    checkpoint_manager = create_checkpoint_manager(save_path, max_to_keep=10)
    if len(os.listdir(save_path)) > 0:
        model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)

    state = model_ckpt['model']
    state = flax.jax_utils.replicate(model_ckpt['model'])
    dl = generator(**dataloader_configs)  # file_path
    finished_steps = model_ckpt['steps']

    with tqdm(total=trainer_configs['total_steps']) as pbar:
        pbar.update(finished_steps)
        for steps in range(finished_steps + 1, 1000000):
            key, train_step_key = jax.random.split(key, num=2)
            train_step_key = shard_prng_key(train_step_key)
            batch = next(dl)

            batch = shard(batch)
            state, metrics = train_step(state, batch, )
            for k, v in metrics.items():
                metrics.update({k: v[0]})

            pbar.set_postfix(metrics)
            pbar.update(1)

            if steps > 100:
                state = update_ema(state, 0.9999)

            if steps % trainer_configs['sample_steps'] == 0:
                save_path=f"{trainer_configs['save_path']}/{steps}.png"

                sample_save_image(state, save_path, steps, batch)
                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}  # 'steps': steps
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)