
import optax
import jax.numpy as jnp

def l2_loss(predictions, target):
    return optax.l2_loss(predictions=predictions, targets=target)


def l1_loss(predictions, target):
    return jnp.abs(target - predictions)
