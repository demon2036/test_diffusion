import einops
import numpy as np
import torch
import torchvision

from data.dataset import MyDataSet, generator
import jax
import jax.numpy as jnp


def get_map(img):
    b, h, w, c = img.shape
    key=jax.random.PRNGKey(0)
    key_lam,key_x,key_y=jax.random.split(key,3)
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

    map = jnp.zeros(img.shape[:3])
    print(map.shape, x1, x2)
    for i in range(b):
        map = map.at[:, x1[i]:x2[i], y1[i]:y2[i]].set(1)

    return map


if __name__ == "__main__":
    image_size = 256
    dl = generator(2, '/home/john/data/s', cache=False, image_size=image_size, repeat=2, dataset=MyDataSet)

    from tqdm import tqdm

    for data in tqdm(dl):
        map = get_map(data)
        img1=data[0]
        img2=data[1]
        map=einops.repeat(map,'b h w -> b h w c',c=3)
        map_1=map[0]
        print(map_1.shape,img1.shape,img2.shape)

        mixed_image=map_1*img1+(1-map_1)*img2

        print(mixed_image.shape)

        all = jnp.stack([img1, img2, mixed_image])

        data = torch.Tensor(np.array(all))

        data = einops.rearrange(data, 'b  h w c->(b ) c h w', )
        torchvision.utils.save_image(data, f'{1}.png', nrow=4)


        break
