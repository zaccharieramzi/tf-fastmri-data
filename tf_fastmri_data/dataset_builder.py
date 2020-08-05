from pathlib import Path

import tensorflow as tf

from .h5 import load_data_from_file as h5_load


def _convert_to_tensors(*args):
    return [tf.convert_to_tensor(arg) for arg in args]

class FastMRIDatasetBuilder:
    def __init__(
            self,
            path,
            mode='train',
            slice_random=False,
            multicoil=False,
            contrast=None,
            af=4,
            shuffle=False,
            seed=0,
            prebuild=True,
            repeat=True,
            n_samples=None,
            prefetch=True,
        ):
        self.path = Path(path)
        self.mode = mode
        self._check_mode()
        self.slice_random = slice_random
        self.multicoil = multicoil
        self.contrast = contrast
        self.af = af
        self.shuffle = shuffle
        self.seed = seed
        self.repeat = repeat
        self.n_samples = n_samples
        self.prefetch = prefetch
        self.files_ds = tf.data.Dataset.list_files(str(path) + '/*.h5', shuffle=False)
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


    def _build_datasets(self):
        self._raw_ds = self.files_ds.map(
            self.load_data,
            num_parallel_calls=self.num_parallel_calls,
        )
        self._filtered_ds = self.raw_ds.map(self.filter_condition)
        self._preprocessed_ds = self.files_ds.map(
            self.preprocessing,
            num_parallel_calls=self.num_parallel_calls,
        )
        if self.n_samples is not None:
            self._preprocessed_ds = self._preprocessed_ds.take(self.n_samples)
        if self.repeat is not None:
            self._preprocessed_ds = self._preprocessed_ds.repeat()
        if self.prefetch is not None:
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

    def preprocessing(self,):
        raise NotImplementedError('You must implement a preprocessing function')


    def _check_mode(self):
        if self.mode not in ['train', 'test']:
            raise ValueError(f'mode must be train or test but is {self.mode}')

    def load_data(self, filename):
        def _load_data(filename):
            filename_str = filename.numpy()
            kspace, image, mask, contrast, af, output_shape = h5_load(
                filename_str,
                slice_random=self.slice_random,
            )
            if self.mode == 'train':
                outputs = (kspace, image, contrast)
            elif self.mode == 'test':
                outputs = (kspace, mask, contrast, af, output_shape)
            return _convert_to_tensors(*outputs)
        if self.mode == 'train':
            output_types = [tf.complex64, tf.float32, tf.string]
        elif self.mode == 'test':
            output_types = [tf.complex64, tf.bool, tf.string, tf.int32, tf.int32]
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
