import time

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.fourier import ortho_ifft2d
from tf_fastmri_data.size_adjustment import crop



dataset_singlecoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True).preprocessed_ds
dataset_multicoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True, multicoil=True).preprocessed_ds
datasets = {
    'Multi coil': dataset_multicoil,
    'Single coil': dataset_singlecoil,
}

for ds_name, dataset in datasets.items():
    # warm-up
    for res in dataset.take(50):
        pass

    start = time.time()
    for res in dataset.take(50):
        pass
    end = time.time()
    print(f'{ds_name} loading (random slice): {(end - start) / 50}s per-file.')
