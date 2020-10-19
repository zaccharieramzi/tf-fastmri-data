from math import pi
import multiprocessing

import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import ifft2d, ifftshift, fftshift


@tf.function
def extract_smaps(kspace, low_freq_percentage=8):
    """Extract raw sensitivity maps for kspaces

    This function will first select a low frequency region in all the kspaces,
    then Fourier invert it, and finally perform a normalisation by the root
    sum-of-square.
    kspace has to be of shape: nslices x ncoils x height x width

    Arguments:
        kspace (tf.Tensor): the kspace whose sensitivity maps you want extracted.
        low_freq_percentage (int): the low frequency region to consider for
            sensitivity maps extraction, given as a percentage of the width of
            the kspace. In fastMRI, it's 8 for an acceleration factor of 4, and
            4 for an acceleration factor of 8. Defaults to 8.

    Returns:
        tf.Tensor: extracted raw sensitivity maps.
    """
    k_shape = tf.shape(kspace)[-2:]
    n_slices = tf.shape(kspace)[0]
    n_coils = tf.shape(kspace)[1]
    n_low_freq = tf.cast(k_shape * low_freq_percentage / 100, tf.int32)
    center_dimension = tf.cast(k_shape / 2, tf.int32)
    low_freq_lower_locations = center_dimension - tf.cast(n_low_freq / 2, tf.int32)
    low_freq_upper_locations = center_dimension + tf.cast(n_low_freq / 2, tf.int32)
    ### Masking strategy
    x_range = tf.range(0, k_shape[0])
    y_range = tf.range(0, k_shape[1])
    X_range, Y_range = tf.meshgrid(x_range, y_range)
    X_mask = tf.logical_and(X_range <= low_freq_upper_locations[0], X_range >= low_freq_lower_locations[0])
    Y_mask = tf.logical_and(Y_range <= low_freq_upper_locations[1], Y_range >= low_freq_lower_locations[1])
    low_freq_mask = tf.transpose(tf.logical_and(X_mask, Y_mask))[None, None, :]
    ###
    low_freq_kspace = kspace * tf.cast(low_freq_mask, kspace.dtype)
    shifted_kspace = ifftshift(low_freq_kspace, axes=[2, 3])
    batched_kspace = tf.reshape(shifted_kspace, (-1, k_shape[0], k_shape[1]))
    batched_coil_image_low_freq_shifted = tf.map_fn(
        ifft2d,
        batched_kspace,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    coil_image_low_freq_shifted = tf.reshape(
        batched_coil_image_low_freq_shifted,
        (n_slices, n_coils, k_shape[0], k_shape[1]),
    )
    coil_image_low_freq = fftshift(coil_image_low_freq_shifted, axes=[2, 3])
    # no need to norm this since they all have the same norm
    low_freq_rss = tf.norm(coil_image_low_freq, axis=1)
    coil_smap = coil_image_low_freq / low_freq_rss[:, None]
    # for now we do not perform background removal based on low_freq_rss
    # could be done with 1D k-means or fixed background_thresh, with tf.where
    return coil_smap


def non_cartesian_extract_smaps(kspace, trajs, dcomp, nufft_back, shape, low_freq_percentage=8):
    def _crop_for_pad(image, shape, im_size):
        to_pad = im_size[-1] - shape[0]
        cropped_image = image[..., to_pad//2:-to_pad//2]
        return cropped_image
    cutoff_freq = low_freq_percentage / 200 * tf.constant(pi)
    # Get the boolean mask for low frequency
    low_freq_bool_mask = tf.math.reduce_all(tf.math.less_equal(tf.abs(trajs[0]), cutoff_freq), axis=0)
    # Obtain the trajectory, kspace and density compensation for low frequency
    low_freq_traj = tf.boolean_mask(trajs, low_freq_bool_mask, axis=2)
    low_freq_kspace = tf.boolean_mask(kspace, low_freq_bool_mask, axis=2)
    low_freq_dcomp = tf.boolean_mask(dcomp, low_freq_bool_mask, axis=1)
    coil_smap = nufft_back(low_freq_kspace * tf.cast(low_freq_dcomp, kspace.dtype), low_freq_traj)
    coil_smap = tf.cond(
            tf.math.greater_equal(shape, coil_smap.shape[-1]),
            lambda: coil_smap,
            lambda: _crop_for_pad(coil_smap, shape, coil_smap.shape),
        )
    low_freq_rss = tf.norm(coil_smap, axis=1)
    coil_smap = coil_smap / low_freq_rss
    return coil_smap
