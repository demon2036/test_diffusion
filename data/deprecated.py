def split_array_into_overlapping_patches(arr, patch_size, stride):
    # Get the array's shape
    batch_size, height, width, num_channels = arr.shape
    num_patches_vertical = (height - patch_size) // stride + 1
    num_patches_horizontal = (width - patch_size) // stride + 1

    # Create an array of indices for extracting patches
    y_indices = stride * jnp.arange(num_patches_vertical)
    x_indices = stride * jnp.arange(num_patches_horizontal)
    yy, xx = jnp.meshgrid(y_indices, x_indices)
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    # Calculate the indices for patches extraction
    y_indices = yy + jnp.arange(patch_size)
    x_indices = xx + jnp.arange(patch_size)

    # Extract the patches using advanced indexing
    patches = arr[:, y_indices[:, :, None], x_indices[:, None, :]]

    return patches


import jax.numpy as jnp
from jax import random, vmap, lax


def random_crop_single(rng_key, image, crop_size):
    image_height, image_width, _ = image.shape
    crop_height, crop_width = crop_size

    if image_height < crop_height or image_width < crop_width:
        raise ValueError("Crop size must be smaller than image dimensions")

    max_y = image_height - crop_height
    max_x = image_width - crop_width

    offset_y = random.randint(rng_key, (), 0, max_y + 1)
    offset_x = random.randint(rng_key, (), 0, max_x + 1)

    cropped_image = lax.dynamic_slice(image, (offset_y, offset_x, 0), (crop_height, crop_width, 3))

    return cropped_image


def random_crop_batch(rng_key, images, crop_size):
    num_images = images.shape[0]

    # Use vmap to apply random_crop_single to each image in the batch
    rng_keys = random.split(rng_key, num_images)
    cropped_images = vmap(random_crop_single, (0, 0, None))(rng_keys, images, crop_size)

    return cropped_images


# Example usage
