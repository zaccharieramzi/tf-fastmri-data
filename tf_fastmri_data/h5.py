import random

import h5py
import ismrmrd


def load_data_from_file(filename, slice_random=False, no_kspace=False):
    with h5py.File(filename, 'r') as h5_obj:
        if no_kspace:
            kspace = None
        else:
            kspace = h5_obj['kspace']
        if 'reconstruction_esc' in h5_obj.keys():
            image = h5_obj['reconstruction_esc']
        elif 'reconstruction_rss' in h5_obj.keys():
            image = h5_obj['reconstruction_rss']
        else:
            image = None
        if slice_random:
            kspace, image = _slice_selection(kspace, image)
        else:
            if kspace is not None:
                kspace = kspace[()]
            if image is not None:
                image = image[()]
        mask = h5_obj.get('mask', None)
        if mask is not None:
            mask = mask[()].astype('bool')
        ismrmrd_header = h5_obj['ismrmrd_header'][()]
        output_shape = _get_output_shape(ismrmrd_header)
        contrast = h5_obj.attrs['acquisition']
        acceleration_factor = h5_obj.attrs.get('acceleration')
        return kspace, image, mask, contrast, acceleration_factor, output_shape

def _slice_selection(kspace, image):
    if kspace is not None:
        base_tensor = kspace
    else:
        base_tensor = image
    i_max = base_tensor.shape[0] - 1
    i_slice = random.randint(0, i_max)
    if kspace is not None:
        kspace = kspace[i_slice]
    if image is not None:
        image = image[i_slice]
    return kspace, image

def _get_output_shape(ismrmrd_header):
    hdr = ismrmrd.xsd.CreateFromDocument(ismrmrd_header)
    enc = hdr.encoding[0]
    enc_size = (enc.encodedSpace.matrixSize.x, enc.encodedSpace.matrixSize.y)
    return enc_size
