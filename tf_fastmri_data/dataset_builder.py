from functools import partial
from pathlib import Path

import tensorflow as tf

from .config import FASTMRI_DATA_DIR, PATHS_MAP
from .h5 import load_data_from_file as load_data_from_file, load_metadata_from_file, load_output_shape_from_file
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from tf_fastmri_data.preprocessing_utils.size_adjustment import crop


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
            complex_image=False,
            batch_size=None,
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
        self.complex_image = complex_image
        self.batch_size = batch_size
        if self.batch_size is not None and not self.slice_random:
            raise ValueError('You can only use batching when selecting one slice')
        if self.slice_random and self.batch_size is None:
            self.batch_size = 1
        self._files = sorted(self.path.glob('*.h5'))
        self.filtered_files = [
            f for f in self._files
            if self.filter_condition(*load_metadata_from_file(f))
        ]
        if not self.filtered_files:
            raise ValueError(
                f'''No files for this contrast ({self.contrast}) and
                acceleration factor ({self.accel_factor})
                found at this path {self.path}'''
            )
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
        if self.complex_image:
            # you can only ask complex image if you ask for kspace
            # for now also available only for knee images (320 x 320)
            self._raw_ds = self._raw_ds.map(
                lambda _, kspace: crop(ortho_ifft2d(kspace), (320, 320))
            )
        if self.brain:
            output_shape_ds = tf.data.Dataset.from_tensor_slices(
                [load_output_shape_from_file(f) for f in self.filtered_files],
            )
            self._raw_ds = tf.data.Dataset.zip(
                (self._raw_ds, output_shape_ds)
            )
            self._raw_ds = self._raw_ds.map(
                lambda tensors, output_shape: (*tensors, output_shape)
            )
        if self.batch_size is not None:
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
    def preprocessed_ds(self):
        if not self.built:
            self._build_datasets()
        return self._preprocessed_ds

    def preprocessing(self, *data_tensors):
        raise NotImplementedError('You must implement a preprocessing function')

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
