import tensorflow as tf

def pad(x, size):
    shape = tf.shape(x)[-1]
    to_pad = size[-1] - shape
    # TODO: adapt this for multicoil
    padded_x = tf.pad(
        x,
        [
            (0, 0),
            (0, 0),
            (to_pad//2, to_pad//2)
        ]
    )
    return padded_x


def crop(x, size):
    shape = tf.shape(x)[-1]
    to_crop = shape - size[-1]
    cropped_x = x[..., to_crop//2:-to_crop//2]
    return cropped_x
