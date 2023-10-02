import chex
import jax
import jax.numpy as jnp


def get_cut_mix_params(keys, w, h):
    key_lam, key_x, key_y = jax.random.split(keys, 3)
    lam = jax.random.beta(key_lam, 1, 1, shape=())
    r = jnp.sqrt(1 - lam)
    w = jnp.int32(w * r)
    h = jnp.int32(h * r)
    x = jax.random.randint(key_x, shape=(), minval=0, maxval=w)
    y = jax.random.randint(key_y, shape=(), minval=0, maxval=h)
    x1 = jnp.clip(x - w // 2, 0, w)
    y1 = jnp.clip(y - h // 2, 0, h)
    x2 = jnp.clip(x + w // 2, 0, w)
    y2 = jnp.clip(y + h // 2, 0, h)

    return x1, y1, x2, y2


def get_cut_mix_label(img, keys):
    b, h, w, c = img.shape
    x1, y1, x2, y2 = get_cut_mix_params(keys, w, h, )

    def create_mask(x1, x2, y1, y2):
        x_indices = slice(x1, x2)
        y_indices = slice(y1, y2)
        x_mask = (jnp.arange(w) >= x_indices.start) & (jnp.arange(w) < x_indices.stop)
        y_mask = (jnp.arange(h) >= y_indices.start) & (jnp.arange(h) < y_indices.stop)
        mask = jnp.outer(y_mask, x_mask)
        return jnp.where(mask, 1.0, 0.0)

    cut_mix_label = create_mask(x1, x2, y1, y2, )
    cut_mix_label=jnp.expand_dims(cut_mix_label,-1)
    return cut_mix_label


def get_mix_up_label(shape, key, alpha=0.2):
    b, h, w, c = shape
    lam = jax.random.beta(key, alpha, alpha, shape=(b, h, w, 1))
    return lam

