import jax.random
from diffusers import FlaxAutoencoderKL, FlaxStableDiffusionPipeline, FlaxStableDiffusionInpaintPipeline
import jax.numpy as jnp
import

if __name__ == "__main__":
    model='sd/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/vae'
    vae, params = FlaxAutoencoderKL.from_pretrained(model, from_pt=True, #subfolder='vae',
                                                    #cache_dir='sd',trust_remote=True
                                                    )
    shape = (1, 4, 16, 16,)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones(shape)
    posterior = vae.apply({'params': params}, x, method=vae.decode)

    print(posterior.sample)
    print(posterior.sample.shape)

    #
    # print(posterior.latent_dist.sample(rng))
    # print(posterior.latent_dist.sample(rng).shape)
