from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import FASTMRI_DATA_DIR, PATHS_MAP
from .h5 import load_data_from_file as h5_load


def _convert_to_tensors(*args):
    return [tf.convert_to_tensor(arg) for arg in args]

class FastMRIDatasetBuilder:
    def __init__(
            self,
            path=None,
            dataset='train',
            brain=False,
            multicoil=False,
            slice_random=False,
            contrast=None,
            af=4,
            shuffle=False,
            seed=0,
            prebuild=True,
            repeat=True,
            n_samples=None,
            prefetch=True,
            no_kspace=False,
        ):
        self.dataset = dataset
        self._check_dataset()
        self.brain = brain
        self.multicoil = multicoil
        if path is None:
            if FASTMRI_DATA_DIR is None:
                raise ValueError('You must specify a path to the data.')
            else:
                path = self._get_path_default()
        self.path = Path(path)
        if self.dataset in ['train', 'val']:
            self.mode = 'train'
        elif self.dataset in ['test']:
            self.mode = 'test'
        self.slice_random = slice_random
        self.contrast = contrast
        self.af = af
        self.shuffle = shuffle
        self.seed = seed
        self.repeat = repeat
        self.n_samples = n_samples
        self.prefetch = prefetch
        self.no_kspace = no_kspace
        self.files_ds = tf.data.Dataset.list_files(str(self.path) + '/*.h5', shuffle=False)
        if self.shuffle:
            self.files_ds = self.files_ds.shuffle(
                buffer_size=1000,
                seed=self.seed,
                reshuffle_each_iteration=False,
            )
        self.num_parallel_calls = tf.data.experimental.AUTOTUNE if self.slice_random else None
        self.built = False
        if prebuild:
            self._build_datasets()

    def _check_dataset(self,):
        if self.dataset not in ['test', 'val', 'train']:
            raise ValueError(
                f'dataset must be train val or test but is {self.dataset}',
            )

    def _get_path_default(self,):
        fastmri_data_dir = Path(FASTMRI_DATA_DIR)
        path_default = fastmri_data_dir / PATHS_MAP[self.multicoil][self.brain][self.dataset]
        return path_default

    def _build_datasets(self):
        self._raw_ds = self.files_ds.map(
            self.load_data,
            num_parallel_calls=self.num_parallel_calls,
        )
        self._filtered_ds = self._raw_ds.filter(self.filter_condition)
        self._preprocessed_ds = self._filtered_ds.map(
            self.preprocessing,
            num_parallel_calls=self.num_parallel_calls,
        )
        if self.n_samples is not None:
            self._preprocessed_ds = self._preprocessed_ds.take(self.n_samples)
        if self.repeat:
            self._preprocessed_ds = self._preprocessed_ds.repeat()
        if self.prefetch:
            self._preprocessed_ds = self._preprocessed_ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.built = True

    @property
    def raw_ds(self):
        if not self.built:
            self._build_datasets()
        return self._raw_ds

    @property
    def filtered_ds(self):
        if not self.built:
            self._build_datasets()
        return self._filtered_ds

    @property
    def preprocessed_ds(self):
        if not self.built:
            self._build_datasets()
        return self._preprocessed_ds

    def preprocessing(self, *data_tensors):
        raise NotImplementedError('You must implement a preprocessing function')

    def load_data(self, filename):
        def _load_data(filename):
            filename_str = filename.numpy()
            kspace, image, mask, contrast, af, output_shape = h5_load(
                filename_str,
                slice_random=self.slice_random,
                no_kspace=self.no_kspace,
            )
            if self.mode == 'train':
                if self.no_kspace:
                    kspace = np.zeros_like(image, dtype=np.complex64)
                outputs = (kspace, image, contrast)
            elif self.mode == 'test':
                outputs = (kspace, mask, contrast, af, output_shape)
            return _convert_to_tensors(*outputs)
        if self.mode == 'train':
            output_types = [tf.complex64, tf.float32, tf.string]
        elif self.mode == 'test':
            output_types = [tf.complex64, tf.bool, tf.string, tf.int64, tf.int32]
        data_tensors = tf.py_function(
            _load_data,
            [filename],
            output_types,
        )
        self._set_tensor_shapes(*data_tensors)
        return data_tensors

    def _set_tensor_shapes(self, *data_tensors):
        if self.mode == 'train':
            kspace, image, contrast = data_tensors
        elif self.mode == 'test':
            kspace, mask, contrast, af, output_shape = data_tensors
        kspace_size = [None] * 3
        if self.multicoil:
            kspace_size.append(None)
        kspace.set_shape(kspace_size)
        if self.mode == 'train':
            image_size = [None] * 3
            image.set_shape(image_size)
            return kspace, image, contrast
        elif self.mode == 'test':
            mask.set_shape([None])
            return kspace, mask, contrast, af, output_shape

    def filter_condition(self, *data_tensors):
        if self.mode == 'train':
            _, _, contrast = data_tensors
            if self.contrast is None:
                return True
            else:
                condition = contrast == self.contrast
                return condition
        elif self.mode == 'test':
            _, _, contrast, af, _ = data_tensors
            condition = af == self.af
            if self.contrast is not None:
                condition = condition and contrast == self.contrast
            return condition
