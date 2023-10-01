import chex
import jax
import jax.numpy as jnp


def get_cut_mix_params(key, w, h, b):
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

    return x1, y1, x2, y2


def get_cut_mix_label(img, key):
    b, h, w, c = img.shape

    x1, y1, x2, y2 = get_cut_mix_params(key, w, h, b)

    x1 = jax.lax.convert_element_type(x1, jnp.int32)
    y1 = jax.lax.convert_element_type(y1, jnp.int32)
    x2 = jax.lax.convert_element_type(x2, jnp.int32)
    y2 = jax.lax.convert_element_type(y2, jnp.int32)

    cut_mix_label = jnp.zeros(img.shape[:3])
    for i in range(b):
        cut_mix_label = cut_mix_label.at[i, x1[i]:x2[i], y1[i]:y2[i]].set(1)
        # cut_mix_label[i, x1[i]:x2[i], y1[i]:y2[i]] = 1
    return cut_mix_label


def cutmix(rng: chex.PRNGKey,
           images: chex.Array,
           labels: chex.Array,
           alpha: float = 1.,
           beta: float = 1.,
           split: int = 1) :
    """Composing two images by inserting a patch into another image."""
    batch_size, height, width, _ = images.shape
    split_batch_size = batch_size // split if split > 1 else batch_size

    # Masking bounding box.
    box_rng, lam_rng, rng = jax.random.split(rng, num=3)
    lam = jax.random.beta(lam_rng, a=alpha, b=beta, shape=())
    cut_rat = jnp.sqrt(1. - lam)
    cut_w = jnp.array(width * cut_rat, dtype=jnp.int32)
    cut_h = jnp.array(height * cut_rat, dtype=jnp.int32)
    box_coords = _random_box(box_rng, height, width, cut_h, cut_w)
    # Adjust lambda.
    lam = 1. - (box_coords[2] * box_coords[3] / (height * width))
    idx = jax.random.permutation(rng, split_batch_size)

    def _cutmix(x, y):
        images_a = x
        images_b = x[idx, :, :, :]
        y = lam * y + (1. - lam) * y[idx, :]
        x = _compose_two_images(images_a, images_b, box_coords)
        return x, y

    if split <= 1:
        return _cutmix(images, labels)

    # Apply CutMix separately on each sub-batch. This reverses the effect of
    # `repeat` in datasets.
    images = einops.rearrange(images, '(b1 b2) ... -> b1 b2 ...', b2=split)
    labels = einops.rearrange(labels, '(b1 b2) ... -> b1 b2 ...', b2=split)
    images, labels = jax.vmap(_cutmix, in_axes=1, out_axes=1)(images, labels)
    images = einops.rearrange(images, 'b1 b2 ... -> (b1 b2) ...', b2=split)
    labels = einops.rearrange(labels, 'b1 b2 ... -> (b1 b2) ...', b2=split)
    return images, labels


def _random_box(rng: chex.PRNGKey,
                height: chex.Numeric,
                width: chex.Numeric,
                cut_h: chex.Array,
                cut_w: chex.Array) -> chex.Array:
    """Sample a random box of shape [cut_h, cut_w]."""
    height_rng, width_rng = jax.random.split(rng)
    i = jax.random.randint(
        height_rng, shape=(), minval=0, maxval=height, dtype=jnp.int32)
    j = jax.random.randint(
        width_rng, shape=(), minval=0, maxval=width, dtype=jnp.int32)
    bby1 = jnp.clip(i - cut_h // 2, 0, height)
    bbx1 = jnp.clip(j - cut_w // 2, 0, width)
    h = jnp.clip(i + cut_h // 2, 0, height) - bby1
    w = jnp.clip(j + cut_w // 2, 0, width) - bbx1
    return jnp.array([bby1, bbx1, h, w])


def _compose_two_images(images: chex.Array,
                        image_permutation: chex.Array,
                        bbox: chex.Array) -> chex.Array:
    """Inserting the second minibatch into the first at the target locations."""

    def _single_compose_two_images(image1, image2):
        height, width, _ = image1.shape
        mask = _window_mask(bbox, (height, width))
        return image1 * (1. - mask) + image2 * mask

    return jax.vmap(_single_compose_two_images)(images, image_permutation)


def _window_mask(destination_box: chex.Array,
                 size) -> jnp.ndarray:
    """Mask a part of the image."""
    height_offset, width_offset, h, w = destination_box
    h_range = jnp.reshape(jnp.arange(size[0]), [size[0], 1, 1])
    w_range = jnp.reshape(jnp.arange(size[1]), [1, size[1], 1])
    return jnp.logical_and(
        jnp.logical_and(height_offset <= h_range,
                        h_range < height_offset + h),
        jnp.logical_and(width_offset <= w_range,
                        w_range < width_offset + w)).astype(jnp.float32)
