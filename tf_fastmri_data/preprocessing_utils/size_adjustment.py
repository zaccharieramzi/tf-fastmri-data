import tensorflow as tf

def pad(x, size):
    shape = tf.shape(x)[-1]
    to_pad = size[-1] - shape
    padding = [(0, 0) for _ in range(len(tf.shape(x)) - 1)]
    padding.append((to_pad//2, to_pad//2))
    padded_x = tf.pad(
        x,
        padding,
    )
    return padded_x


def crop(x, size):
    shape = tf.shape(x)[-2:]
    to_crop = shape - size
    cropped_x = x[
        ...,
        to_crop[0]//2:shape[0]-to_crop[0]//2,
        to_crop[1]//2:shape[1]-to_crop[1]//2,
    ]
    return cropped_x
