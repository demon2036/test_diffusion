import numpy as np

# def pyramid_noise_like(x, discount=0.9):
#   b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
#   u = nn.Upsample(size=(w, h), mode='bilinear')
#   noise = torch.randn_like(x)
#   for i in range(10):
#     r = random.random()*2+2 # Rather than always going 2x,
#     w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
#     noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
#     if w==1 or h==1: break # Lowest resolution is 1x1
#   return noise/noise.std() # Scaled back to roughly unit variance

#

import jax.numpy as jnp
import torch

from gaussian_test import test
from torch_diffusion import GaussianDiffusion,Unet

from diffusers import UNet2DModel

def show(l1,l2):
    for a,b in zip(l1,l2):
        print(np.max(np.array(a)-np.array(b)))

if __name__=="__main__":
    c=test( loss='l2', image_size=64, timesteps=1000, sampling_timesteps=999, beta_schedule='linear')

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        beta_schedule='linear',
        objective='pred_noise',
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )



    # x1=c.posterior_mean_coef2
    # x2=diffusion.posterior_mean_coef2
    # print(np.max(np.array(x1)-np.array(x2)))
    shape=((1,4,4,3))
    time=100
    img1=jnp.ones(shape=shape)
    noise1=jnp.zeros_like(img1)
    t1=jnp.full((1,),time)
    x1=c.q_sample(img1,t1,noise1)

    a1,b1,c1=c.q_posterior(img1,x1,t1)

    device='cpu'
    img2=torch.ones(shape,device=device)
    t2=torch.full((1,),time,dtype=torch.long,device=device)
    noise2=torch.zeros_like(img2,device=device)
    x2=diffusion.q_sample(img2,t2,noise2)
    a2, b2, c2 = diffusion.q_posterior(img2, x2, t2)

    show([ a1,b1,c1],[a2, b2, c2 ])

