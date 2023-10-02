import jax
import jax.numpy as jnp
import jax
import jax.numpy as np
from jax import random, jit
import tensorflow_models as tfm
import tensorflow as tf

cutmix = tfm.vision.augment.MixupAndCutmix(num_classes=1000)

shape = (2, 224, 224, 3)
z = tf.zeros(shape)
label = tf.zeros((2,))
cutmix(z, label)
