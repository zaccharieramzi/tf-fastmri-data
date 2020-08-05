import tensorflow as tf
from tfkbnufft import kbnufft_forward


def _pad_for_nufft(image, im_size):
    shape = tf.shape(image)[-1]
    to_pad = im_size[-1] - shape
    padded_image = tf.pad(
        image,
        [
            (0, 0),
            (0, 0),
            (0, 0),
            (to_pad//2, to_pad//2)
        ]
    )
    return padded_image


def _crop_for_nufft(image, im_size):
    shape = tf.shape(image)[-1]
    to_crop = shape - im_size[-1]
    cropped_image = image[..., to_crop//2:-to_crop//2]
    return cropped_image

def nufft(nufft_ob, image, ktraj, image_size=None):
    forward_op = kbnufft_forward(nufft_ob._extract_nufft_interpob())
    shape = tf.shape(image)[-1]
    if image_size is not None:
        image_adapted = tf.cond(
            tf.math.greater(shape, image_size[-1]),
            lambda: _crop_for_nufft(image, image_size),
            lambda: _pad_for_nufft(image, image_size),
        )
    else:
        image_adapted = image
    kspace = forward_op(image_adapted, ktraj)
    return kspace
