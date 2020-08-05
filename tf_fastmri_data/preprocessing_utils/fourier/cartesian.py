import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import ifft2d, ifftshift, fftshift


def ortho_ifft2d(kspace):
    axes = [len(kspace.shape) - 2, len(kspace.shape) - 1]
    scaling_norm = tf.cast(tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')), kspace.dtype)
    return scaling_norm * fftshift(ifft2d(ifftshift(kspace, axes=axes)), axes=axes)
