import time

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from tf_fastmri_data.preprocessing_utils.size_adjustment import crop


dataset_singlecoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True)
dataset_multicoil = CartesianFastMRIDatasetBuilder(dataset='val', slice_random=True, multicoil=True)
datasets = {
    'Multi coil': dataset_multicoil,
    'Single coil': dataset_singlecoil,
}
### LOADING BENCHMARK
N_WARMUP = 50
N_LOADS = 100
#
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


def h5_load(file):
    with h5py.File(file, 'r') as h5_obj:
        kspace = h5_obj['kspace'][0]
        image = h5_obj['reconstruction_rss'][0]
        mask = None
    kspace = kspace[None, ..., None]
    image = image[None, ..., None]
    return (kspace, mask), image

for ds_name, dataset in datasets.items():
    files = dataset.filtered_files
    n_files = len(files)
    start = time.time()
    for file in tqdm(files, total=n_files):
        h5_load(file)
    end = time.time()
    print(f'{ds_name} with h5py loading (random slice, without preprocessing): {(end - start) / n_files}s per-file.')


### TRAINING BENCHMARK (single coil only)
class SimpleModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_conv = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
        )
        self.final_conv = Conv2D(
            filters=1,
            kernel_size=3,
            activation=None,
            padding='same',
        )

    def call(self, inputs):
        kspace, _mask = inputs
        zfilled_inputs = tf.abs(ortho_ifft2d(kspace[..., 0]))
        zfilled_inputs = crop(zfilled_inputs, (320, 320))
        zfilled_inputs = zfilled_inputs[..., None]
        outputs = zfilled_inputs
        outputs = self.first_conv(outputs)
        outputs = self.final_conv(outputs)
        outputs = zfilled_inputs + outputs
        return outputs


model = SimpleModel()
n_epochs = 100
model.compile(loss='mse', optimizer='sgd')
start = time.time()
model.run_eagerly = True
model.fit(
    dataset_singlecoil.preprocessed_ds.take(n_epochs),
    callbacks=[TensorBoard(write_graph=False, profile_batch='10,15')]
)
end = time.time()
print(f'Single coil training with tfio loading: {(end - start) / n_epochs}s per-step.')
