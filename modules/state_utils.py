import optax
from flax.training import train_state
from typing import Any
import jax.numpy as jnp

from modules.utils import get_obj_from_str


class EMATrainState(train_state.TrainState):
    batch_stats:Any=None
    ema_params: Any = None


def create_state(rng, model_cls, input_shapes, train_state, print_model=True, optimizer_dict=None,
                 model_kwargs=None, ):
    model = model_cls(**model_kwargs)
    inputs = list(map(lambda shape: jnp.empty(shape), input_shapes))


    if print_model:
        print(model.tabulate(rng, *inputs,z_rng=rng, depth=2, console_kwargs={'width': 200}))

    variables = model.init(rng, *inputs,z_rng=rng)
    optimizer = get_obj_from_str(optimizer_dict['optimizer'])

    args=tuple()
    if 'clip_norm' in optimizer_dict and optimizer_dict['clip_norm']:
        args+=( optax.clip_by_global_norm(1)  ,)

    args+=(optimizer(**optimizer_dict['optimizer_configs']),)
    tx = optax.chain(
        *args
    )
    return train_state.create(apply_fn=model.apply,
                              params=variables['params'],
                              tx=tx,
                              batch_stats=variables['batch_stats'] if 'batch_stats' in variables.keys() else None,
                              ema_params=variables['params'])