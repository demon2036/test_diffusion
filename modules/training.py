import jax


def accumulate_gradient(loss_and_grad_fn, params, images, accum_steps):

    print(f'accum_steps:{accum_steps}')

    """Accumulate gradient over multiple steps to save on memory."""
    # See (internal link) for details and experiments.
    if accum_steps and accum_steps > 1:
        assert images.shape[0] % accum_steps == 0, (
            f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}")
        step_size = images.shape[0] // accum_steps
        l, g = loss_and_grad_fn(params, images[:step_size])

        def acc_grad_and_loss(i, l_and_g):
            imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                         (step_size,) + images.shape[1:])

            li, gi = loss_and_grad_fn(params, imgs, )
            l, g = l_and_g

            return jax.tree_util.tree_map(lambda x, y: x + y, l, li), jax.tree_util.tree_map(lambda x, y: x + y, g, gi)

            #return l + li, jax.tree_util.tree_map(lambda x, y: x + y, g, gi)

        l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
        return jax.tree_util.tree_map(lambda x: x / accum_steps, (l, g))
    else:
        return loss_and_grad_fn(params, images)


"""
def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
 
  # See (internal link) for details and experiments.
  if accum_steps and accum_steps > 1:
    assert images.shape[0] % accum_steps == 0, (
        f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}")
    step_size = images.shape[0] // accum_steps
    l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])
    def acc_grad_and_loss(i, l_and_g):
      imgs = jax.lax.dynamic_slice(images, (i*step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i*step_size, 0),
                                   (step_size, labels.shape[1]))
      li, gi = loss_and_grad_fn(params, imgs, lbls)
      l, g = l_and_g
      return (l + li, jax.tree_util.tree_map(lambda x, y: x + y, g, gi))
    l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
    return jax.tree_util.tree_map(lambda x: x / accum_steps, (l, g))
  else:
    return loss_and_grad_fn(params, images, labels)

"""
