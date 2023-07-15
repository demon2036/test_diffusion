import omegaconf
import yaml
import os
import json

state = flax.jax_utils.replicate(model_ckpt['model'])
time_steps = [20, 25, 35, 50, 75, 100, 200, 250, 500, 1000]
for time in time_steps:
    c.sampling_timesteps = time
    sample_save_image(key, c, time, state)
