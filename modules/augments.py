import jax
import jax.numpy as jnp


def get_cut_mix_label(img,key):
    b, h, w, c = img.shape
    key_lam, key_x, key_y = jax.random.split(key, 3)
    lam = jax.random.beta(key_lam, 1, 1, shape=(b,))
    r = jnp.sqrt(1 - lam)
    w = jnp.int32(w * r)
    h = jnp.int32(h * r)
    x = jax.random.randint(key_x, shape=(b,), minval=0, maxval=w)
    y = jax.random.randint(key_y, shape=(b,), minval=0, maxval=h)
    x1 = jnp.clip(x - w // 2, 0, w)
    y1 = jnp.clip(y - h // 2, 0, h)
    x2 = jnp.clip(x + w // 2, 0, w)
    y2 = jnp.clip(y + h // 2, 0, h)

    cut_mix_label = jnp.zeros(img.shape[:3])
    for i in range(b):
        cut_mix_label = cut_mix_label.at[:, x1[i]:x2[i], y1[i]:y2[i]].set(1)
    return cut_mix_label
