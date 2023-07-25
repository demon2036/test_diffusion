from dataset import generator
from ldm.autoencoder import *
from discriminator import EMATrainState

@partial(jax.pmap, axis_name='batch')  # static_broadcasted_argnums=(3),
def train_step(state: EMATrainState, batch, ):
    def loss_fn(params):
        reconstruct = state.apply_fn({'params': params}, batch)
        loss = l1_loss(reconstruct, batch)
        return loss.mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metric = {"loss": loss}
    return new_state, metric

def create_state(rng, model_cls, input_shape, optimizer, train_state=EMATrainState, print_model=True,
                 optimizer_kwargs=None, model_kwargs=None):
    model = model_cls(**model_kwargs)
    if print_model:
        print(model.tabulate(rng, jnp.empty(input_shape), depth=2,
                             console_kwargs={'width': 200}))
    variables = model.init(rng, jnp.empty(input_shape))

    if optimizer == 'AdamW':
        optimizer = optax.adamw
    elif optimizer == "Lion":
        optimizer = optax.lion
    else:
        assert "some thing is wrong"

    tx = optax.chain(
        optax.clip_by_global_norm(1),
        optimizer(**optimizer_kwargs)
    )
    return train_state.create(apply_fn=model.apply, params=variables['params'], tx=tx,
                              ema_params=variables['params'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='./configs/AutoEncoder/test.yaml')
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
                save_path=f"{trainer_configs['save_path']}"
                sample_save_image_autoencoder(state, save_path, steps, batch)
                unreplicate_state = flax.jax_utils.unreplicate(state)
                model_ckpt = {'model': unreplicate_state, 'steps': steps}  # 'steps': steps
                save_args = orbax_utils.save_args_from_target(model_ckpt)
                checkpoint_manager.save(steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)