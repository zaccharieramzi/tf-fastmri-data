from functools import partial
from pathlib import Path

import tensorflow as tf

from .config import FASTMRI_DATA_DIR, PATHS_MAP
from .h5 import load_data_from_file as load_data_from_file, load_metadata_from_file
from tf_fastmri_data.preprocessing_utils.size_adjustment import pad, crop

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
            batch_size=None,
            kspace_size=(640, 372),
            # WARNING: for now not working
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
        self._no_kspace = no_kspace
        self.batch_size = batch_size
        self.kspace_size = kspace_size
        if self.batch_size is not None and not self.slice_random:
            raise ValueError('You can only use batching when selecting one slice')
        if self.slice_random and self.batch_size is None:
            self.batch_size = 1
        # self.set_kspace_same_size()
        self._files = sorted(self.path.glob('*.h5'))
        self.filtered_files = [
            f for f in self._files
            if self.filter_condition(*load_metadata_from_file(f))
        ]
        self.files_ds = tf.data.Dataset.from_tensor_slices(
            [str(f) for f in self.filtered_files],
        )
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
            partial(
                load_data_from_file,
                slice_random=self.slice_random,
                no_kspace=self.no_kspace,
                multicoil=self.multicoil,
                mode=self.mode,
            ),
            num_parallel_calls=self.num_parallel_calls,
        )
        # TODO: add the output_shape for brain ds
        if self.batch_size is not None:
            # if self.same_size_kspace:
            #     self._filtered_ds = self._filtered_ds.map(
            #         self.pad_crop_kspace,
            #         num_parallel_calls=self.num_parallel_calls,
            #     )
            self._raw_ds = self._raw_ds.batch(self.batch_size)
        self._preprocessed_ds = self._raw_ds.map(
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

    @property
    def no_kspace(self):
        return self._no_kspace

    def set_kspace_same_size(self):
        self.same_size_kspace = self.batch_size is None or (self.batch_size > 1 and not self._no_kspace)

    @no_kspace.setter
    def no_kspace(self, val):
        self._no_kspace = val
        self.set_kspace_same_size()

    def preprocessing(self, *data_tensors):
        raise NotImplementedError('You must implement a preprocessing function')

    def _set_tensor_shapes(self, *data_tensors):
        if self.mode == 'train':
            kspace, image, contrast = data_tensors
        elif self.mode == 'test':
            kspace, mask, contrast, af, output_shape = data_tensors
        kspace_size = [None] * 2
        if not self.slice_random:
            kspace_size.append(None)
        if self.multicoil:
            kspace_size.append(None)
        kspace.set_shape(kspace_size)
        if self.mode == 'train':
            image_size = [None] * 2
            if not self.slice_random:
                image_size.append(None)
            image.set_shape(image_size)
            return kspace, image, contrast
        elif self.mode == 'test':
            mask.set_shape([None])
            return kspace, mask, contrast, af, output_shape

    def pad_crop_kspace(self, *data_tensors):
        kspace, *others = data_tensors
        # NOTE: for now only doing it for the last dimension
        shape = tf.shape(kspace)[-1]
        kspace_adapted = tf.cond(
            tf.math.greater(shape, self.kspace_size[-1]),
            lambda: crop(kspace, self.kspace_size),
            lambda: pad(kspace, self.kspace_size),
        )
        outputs = [kspace_adapted] + others
        return outputs

    def filter_condition(self, contrast, af=None):
        if self.mode == 'train':
            if self.contrast is None:
                return True
            else:
                condition = contrast == self.contrast
                return condition
        elif self.mode == 'test':
            condition = af == self.af
            if self.contrast is not None:
                condition = condition and contrast == self.contrast
            return condition
