from pathlib import  Path

import tensorflow as tf

from tf_fastmri_data.config import FASTMRI_DATA_DIR, PATHS_MAP
from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.extract_smaps import extract_smaps
from tf_fastmri_data.preprocessing_utils.masking import mask_random, mask_equidistant, mask_reshaping_and_casting
from tf_fastmri_data.preprocessing_utils.scaling import scale_tensors


class CartesianFastMRIDatasetBuilder(FastMRIDatasetBuilder):
    def __init__(
            self,
            path=None,
            dataset='train',
            brain=False,
            multicoil=False,
            mask_mode=None,
            scale_factor=1e6,
            output_shape_spec=None,
            **kwargs,
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
        if mask_mode is None:
            if self.brain:
                self.mask_mode = 'equidistant'
            else:
                self.mask_mode = 'random'
        else:
            self.mask_mode = mask_mode
        self._check_mask_mode()
        self.scale_factor = scale_factor
        if output_shape_spec is None:
            self.output_shape_spec = brain
        else:
            self.output_shape_spec = output_shape_spec
        if self.dataset in ['train', 'val']:
            self.mode = 'train'
            kwargs.update(prefetch=True)
        elif self.dataset in ['test']:
            self.mode = 'test'
            kwargs.update(repeat=False, prefetch=False)
        super(CartesianFastMRIDatasetBuilder, self).__init__(
            path=path,
            mode=self.mode,
            multicoil=self.multicoil
            **kwargs,
        )

    def _check_mask_mode(self,):
        if self.mask_mode not in ['random', 'equidistant']:
            raise ValueError(
                f'mask_mode must be random or equidistant but is {self.mask_mode}',
            )

    def _check_dataset(self,):
        if self.dataset not in ['test', 'val', 'train']:
            raise ValueError(
                f'dataset must be train val or test but is {self.dataset}',
            )

    def _get_path_default(self,):
        fastmri_data_dir = Path(FASTMRI_DATA_DIR)
        path_default = fastmri_data_dir / PATHS_MAP[self.brain][self.multicoil][self.dataset]
        return path_default


    def gen_mask(self, kspace):
        if self.mask_mode == 'random':
            mask_function = mask_random
        elif self.mask_mode == 'equidistant':
            mask_function = mask_equidistant
        mask = mask_function(
            kspace,
            accel_factor=self.af,
            multicoil=self.multicoil,
        )
        return mask

    def _preprocessing_train(self, kspace, image, _contrast):
        mask = self.gen_mask(kspace)
        kspace = tf.cast(mask, kspace.dtype) * kspace
        kspace, image = scale_tensors(kspace, image, scale_factor=self.scale_factor)
        kspace = kspace[..., None]
        image = image[..., None]
        model_inputs = (kspace, mask)
        if self.multicoil:
            smaps = extract_smaps(kspace[..., 0], low_freq_percentage=32//self.af)
            model_inputs += (smaps,)
        if self.output_shape_spec:
            output_shape = tf.shape(image)[1:][None, :]
            output_shape = tf.tile(output_shape, [tf.shape(image)[0], 1])
            model_inputs += (output_shape,)
        return model_inputs, image

    def _preprocessing_test(self, kspace, mask, _contrast, _af, output_shape):
        kspace = scale_tensors(kspace, scale_factor=self.scale_factor)
        kspace = kspace[..., None]
        mask = mask_reshaping_and_casting(mask, tf.shape(kspace[..., 0]), multicoil=self.multicoil)
        model_inputs = (kspace, mask)
        if self.multicoil:
            smaps = extract_smaps(kspace[..., 0], low_freq_percentage=32//self.af)
            model_inputs += (smaps,)
        if self.output_shape_spec:
            output_shape = output_shape[None, :]
            output_shape = tf.tile(output_shape, [tf.shape(kspace)[0], 1])
            model_inputs += (output_shape,)
        return model_inputs

    def preprocessing(self, *data_tensors):
        if self.mode == 'train':
            preproc_fun = self._preprocessing_train
        elif self.mode == 'test':
            preproc_fun = self._preprocessing_test
        preprocessing_outputs = preproc_fun(*data_tensors)
        return preprocessing_outputs
