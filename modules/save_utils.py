import numpy as np
import torch
from torchvision.utils import save_image


def save_image_from_jax(x,save_name):
    all_image = np.array(x)
    all_image = torch.Tensor(all_image)
    save_image(all_image, f'{save_name}.png')