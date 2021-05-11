import tensorflow as tf
from tfkbnufft import kbnufft_forward

from ..crop import adjust_image_size


def nufft(nufft_ob, image, ktraj, image_size=None, multicoil=True):
    if image_size is not None:
        image = adjust_image_size(
            image,
            image_size,
            multicoil=multicoil,
        )
    forward_op = kbnufft_forward(nufft_ob._extract_nufft_interpob(), multiprocessing=True)
    shape = tf.shape(image)[-1]
    kspace = forward_op(image, ktraj)
    return kspace
