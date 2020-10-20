import multiprocessing

import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import ifft2d, ifftshift, fftshift


def ortho_ifft2d(kspace):
    axes = [len(kspace.shape) - 2, len(kspace.shape) - 1]
    scaling_norm = tf.cast(tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')), kspace.dtype)
    if len(kspace.shape) == 4:
        # multicoil case
        ncoils = tf.shape(kspace)[1]
    n_slices = tf.shape(kspace)[0]
    k_shape_x = tf.shape(kspace)[-2]
    k_shape_y = tf.shape(kspace)[-1]
    shifted_kspace = ifftshift(kspace, axes=axes)
    batched_shifted_kspace = tf.reshape(shifted_kspace, (-1, k_shape_x, k_shape_y))
    batched_shifted_image = tf.map_fn(
        ifft2d,
        batched_shifted_kspace,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    if len(kspace.shape) == 4:
        # multicoil case
        image_shape = [n_slices, ncoils, k_shape_x, k_shape_y]
    elif len(kspace.shape) == 3:
        image_shape = [n_slices, k_shape_x, k_shape_y]
    else:
        image_shape = [k_shape_x, k_shape_y]
    shifted_image = tf.reshape(batched_shifted_image, image_shape)
    image = fftshift(shifted_image, axes=axes)
    return scaling_norm * image
