import time

import h5py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from tf_fastmri_data.preprocessing_utils.size_adjustment import crop



dataset_singlecoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True)
dataset_multicoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True, multicoil=True)
datasets = {
    'Multi coil': dataset_multicoil,
    'Single coil': dataset_singlecoil,
}

for ds_name, dataset in datasets.items():
    ds = dataset.preprocessed_ds
    # warm-up
    for res in ds.take(50):
        pass

    start = time.time()
    for res in ds.take(50):
        pass
    end = time.time()
    print(f'{ds_name} with tfio loading (random slice): {(end - start) / 50}s per-file.')

for ds_name, dataset in datasets.items():
    start = time.time()
    for file in dataset.filtered_files[:50]:
        with h5py.File(file, 'r') as h5_obj:
            kspace = h5_obj['kspace'][0]
    end = time.time()
    print(f'{ds_name} with h5py loading (random slice, only k-space without preprocessing): {(end - start) / 50}s per-file.')


class SimpleModel(Model):
    def __init__(self, arg):
        pass
