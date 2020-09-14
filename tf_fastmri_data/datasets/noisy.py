import tensorflow as tf

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.scaling import scale_tensors


class NoisyFastMRIDatasetBuilder(FastMRIDatasetBuilder):
    def __init__(
            self,
            dataset='train',
            brain=False,
            scale_factor=1e6,
            noise_power_spec=30,
            noise_input=True,
            noise_mode='uniform',
            residual_learning=False,
            batching=False,
            slice_random=False,
            **kwargs,
        ):
        self.dataset = dataset
        if self.dataset in ['train', 'val']:
            kwargs.update(prefetch=True)
        elif self.dataset in ['test']:
            kwargs.update(repeat=False, prefetch=False)
        self.brain = brain
        self.scale_factor = scale_factor
        self.noise_power_spec = noise_power_spec
        self.noise_input = noise_input
        self.noise_mode = noise_mode
        self.residual_learning = residual_learning
        self.batching = batching
        self.slice_random = slice_random
        if self.batching and not self.slice_random:
            raise ValueError('You can only use batching when selecting one slice')
        super(NoisyFastMRIDatasetBuilder, self).__init__(
            dataset=self.dataset,
            brain=self.brain,
            slice_random=self.slice_random,
            no_kspace=True,
            **kwargs,
        )
        if self.mode == 'test':
            raise NotImplementedError('Noisy dataset only works for train/val')

    def _preprocessing_train(self, _kspace, image, _contrast):
        image = scale_tensors(image, scale_factor=self.scale_factor)[0]
        image = image[..., None]
        if self.batching:
            image = image[0]
        noise_power = self.draw_noise_power(batch_size=tf.shape(image)[0])
        noise = tf.random.normal(
            shape=tf.shape(image),
            mean=0.0,
            stddev=1.0,
            dtype=image.dtype,
        )
        if not self.batching
            noise_power_bdcast = noise_power[:, None, None, None]
        else:
            noise_power_bdcast = noise_power
        noise = noise *
        image_noisy = image + noise
        model_inputs = (image_noisy,)
        if self.noise_input:
            model_inputs += (noise_power,)
        if self.residual_learning:
            model_outputs = noise
        else:
            model_outputs = image
        return model_inputs, model_outputs

    def _draw_gaussian_noise_power(self, batch_size):
        noise_power = tf.random.normal(
            shape=(batch_size,),
            mean=0.0,
            stddev=self.noise_power_spec,
        )
        return noise_power

    def _draw_uniform_noise_power(self, batch_size):
        if isinstance(self.noise_power_spec, (int, float)):
            noise_power = (self.noise_power_spec, self.noise_power_spec)
        else:
            noise_power = self.noise_power_spec
        noise_power = tf.random.uniform(
            (batch_size,),
            minval=noise_power[0],
            maxval=noise_power[1],
        )
        return noise_power

    def draw_noise_power(self, batch_size):
        if self.noise_mode == 'uniform':
            draw_func = self._draw_uniform_noise_power
        else:
            draw_func = self._draw_gaussian_noise_power
        noise_power = draw_func(batch_size)
        return noise_power

    def preprocessing(self, *data_tensors):
        preprocessing_outputs = self._preprocessing_train(*data_tensors)
        return preprocessing_outputs
