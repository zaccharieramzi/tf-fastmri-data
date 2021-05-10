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

def adjust_image_size(image, target_image_size, multicoil=False):
    """Resize an image to a target size using centered cropping or padding

    Args:
        - image (tf.Tensor): an image with dimensions (n_slices, n_coils, height, width)
        - target_image_size (list or tuple): the height and width for the output image
        - multicoil (bool): defaults to False. Whether the image has a coil dimension.

    Returns:
        - tf.Tensor: a size-adjusted image
    """
    height = tf.shape(image)[-2]
    width = tf.shape(image)[-1]
    n_slices = tf.shape(image)[0]
    transpose_axis = [1, 2, 0] if not multicoil else [2, 3, 0, 1]
    transposed_image = tf.transpose(image, transpose_axis)
    reshaped_image = tf.reshape(transposed_image, [height, width, -1])  # 3D tensors accepted
    # with channels dimension last
    target_height = target_image_size[0]
    target_width = target_image_size[1]
    padded_image = tf.image.resize_with_crop_or_pad(
        reshaped_image,
        target_height,
        target_width,
    )
    if multicoil:
        final_shape = [target_height, target_width, n_slices, -1]
    else:
        final_shape = [target_height, target_width, n_slices]
    reshaped_padded_image = tf.reshape(padded_image, final_shape)
    transpose_axis = [2, 0, 1] if not multicoil else [2, 3, 0, 1]
    transpose_padded_image = tf.transpose(reshaped_padded_image, transpose_axis)
    return transpose_padded_image
