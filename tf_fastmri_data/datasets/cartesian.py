import tensorflow as tf
import tensorflow_io as tfio

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.extract_smaps import extract_smaps
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from tf_fastmri_data.preprocessing_utils.masking import mask_random, mask_equidistant, mask_reshaping_and_casting
from tf_fastmri_data.preprocessing_utils.scaling import scale_tensors


class CartesianFastMRIDatasetBuilder(FastMRIDatasetBuilder):
    def __init__(
            self,
            dataset='train',
            brain=False,
            mask_mode=None,
            scale_factor=1e6,
            output_shape_spec=None,
            **kwargs,
        ):
        self.dataset = dataset
        if self.dataset in ['train', 'val']:
            kwargs.update(prefetch=True)
        elif self.dataset in ['test']:
            if tuple(int(v) for v in tfio.__version__.split('.')) <= (0, 15, 0):
                raise ValueError(
                    '''Test cartesian dataset is not available for
                    tfio under 1.5.0 because it cannot handle boolean data,
                    see https://github.com/tensorflow/io/issues/1144'''
                )
            kwargs.update(repeat=False, prefetch=False)
        self.brain = brain
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
        super(CartesianFastMRIDatasetBuilder, self).__init__(
            dataset=self.dataset,
            brain=self.brain,
            **kwargs,
        )

    def _check_mask_mode(self,):
        if self.mask_mode not in ['random', 'equidistant']:
            raise ValueError(
                f'mask_mode must be random or equidistant but is {self.mask_mode}',
            )

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

    def _preprocessing_train(self, image, kspace, output_shape=None):
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

    def _preprocessing_test(self, mask, kspace, output_shape=None):
        (kspace,) = scale_tensors(kspace, scale_factor=self.scale_factor)
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
