import jax.numpy as jnp

def create_mask(x1, x2, y1, y2, h, w):
    x_indices = slice(x1, x2)
    y_indices = slice(y1, y2)

    x_mask = (jnp.arange(w) >= x_indices.start) & (jnp.arange(w) < x_indices.stop)
    y_mask = (jnp.arange(h) >= y_indices.start) & (jnp.arange(h) < y_indices.stop)

    print(x_mask)

    mask = jnp.outer(y_mask, x_mask)

    return mask

# Example usage:
x1 = 1
x2 = 4
y1 = 2
y2 = 5
h = 5
w = 5

mask = create_mask(x1, x2, y1, y2, h, w)
print(mask)