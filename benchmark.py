import time

import h5py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from tf_fastmri_data.preprocessing_utils.size_adjustment import crop


N_WARMUP = 50
N_LOADS = 100
dataset_singlecoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True)
dataset_multicoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True, multicoil=True)
datasets = {
    'Multi coil': dataset_multicoil,
    'Single coil': dataset_singlecoil,
}

for ds_name, dataset in datasets.items():
    ds = dataset.preprocessed_ds
    # warm-up
    for res in ds.take(N_WARMUP):
        pass

    start = time.time()
    for res in tqdm(ds.take(N_LOADS), total=N_LOADS):
        pass
    end = time.time()
    print(f'{ds_name} with tfio loading (random slice): {(end - start) / N_LOADS}s per-file.')

for ds_name, dataset in datasets.items():
    files = dataset.filtered_files
    n_files = len(files)
    start = time.time()
    for file in tqdm(files, total=n_files):
        with h5py.File(file, 'r') as h5_obj:
            kspace = h5_obj['kspace'][0]
    end = time.time()
    print(f'{ds_name} with h5py loading (random slice, only k-space without preprocessing): {(end - start) / n_files}s per-file.')


class SimpleModel(Model):
    def __init__(self, arg):
        pass
