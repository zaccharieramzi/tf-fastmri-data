import h5py
import ismrmrd
import tensorflow as tf
import tensorflow_io as tfio


def load_data_from_file(fpath, slice_random=False, no_kspace=False, multicoil=False, mode='train'):
    if multicoil:
        image_name = '/reconstruction_rss'
        kspace_shape = tuple([None]*4)
    else:
        image_name = '/reconstruction_esc'
        kspace_shape = tuple([None]*3)
    image_shape = tuple([None]*3)
    if slice_random:
        kspace_shape = kspace_shape[1:]
        image_shape = image_shape[1:]
    mask_shape = (None,)
    kspace_name = '/kspace'
    mask_name = '/mask'
    spec = {}
    if mode == 'train':
        spec.update({
            image_name: tf.TensorSpec(shape=image_shape, dtype=tf.float32),
        })
    else:
        spec.update({
            mask_name: tf.TensorSpec(shape=mask_shape, dtype=tf.bool),
        })
    if not no_kspace:
        spec.update({
            kspace_name: tf.TensorSpec(shape=kspace_shape, dtype=tf.complex64),
        })
    h5_tensors = tfio.IOTensor.from_hdf5(fpath, spec=spec)
    if no_kspace:
        main_tensor = image_name
    else:
        main_tensor = kspace_name
    h5_main = h5_tensors(main_tensor)
    n_slices = h5_main.shape[0]
    if slice_random:
        i_slice = tf.random.uniform(
            shape=(),
            minval=0,
            maxval=n_slices,
            dtype=tf.int64,
        )
        slices = (i_slice, i_slice + 1)
    else:
        slices = (0, n_slices)
    if mode == 'train':
        image = h5_tensors(image_name)[slices[0]:slices[1]]
        if slice_random:
            image = tf.squeeze(image, axis=0)
        image.set_shape(image_shape)
        outputs = [image]
    else:
        mask = h5_tensors(mask_name)[:]
        mask.set_shape(mask_shape)
        outputs = [mask]
    if not no_kspace:
        kspace = h5_tensors(kspace_name)[slices[0]:slices[1]]
        if slice_random:
            kspace = tf.squeeze(kspace, axis=0)
        kspace.set_shape(kspace_shape)
        outputs.append(kspace)
    return outputs

def load_metadata_from_file(filename):
    with h5py.File(filename, 'r') as h5_obj:
        contrast = h5_obj.attrs['acquisition']
        acceleration_factor = h5_obj.attrs.get('acceleration')
        return contrast, acceleration_factor

def load_output_shape_from_file(filename):
    with h5py.File(filename, 'r') as h5_obj:
        ismrmrd_header = h5_obj['ismrmrd_header'][()]
        output_shape = _get_output_shape(ismrmrd_header)
        return output_shape

def _get_output_shape(ismrmrd_header):
    hdr = ismrmrd.xsd.CreateFromDocument(ismrmrd_header)
    enc = hdr.encoding[0]
    enc_size = (enc.encodedSpace.matrixSize.x, enc.encodedSpace.matrixSize.y)
    return enc_size
