import tensorflow as tf

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.scaling import scale_tensors


class NoisyFastMRIDatasetBuilder(FastMRIDatasetBuilder):
    def __init__(
            self,
            dataset='train',
            brain=False,
            scale_factor=1e6,
            noise_power=30,
            noise_input=True,
            residual_learning=False,
            **kwargs,
        ):
        self.dataset = dataset
        if self.dataset in ['train', 'val']:
            kwargs.update(prefetch=True)
        elif self.dataset in ['test']:
            kwargs.update(repeat=False, prefetch=False)
        self.brain = brain
        self.scale_factor = scale_factor
        if isinstance(noise_power, (int, float)):
            noise_power = (noise_power, noise_power)
        self.noise_power = noise_power
        self.noise_input = noise_input
        self.residual_learning = residual_learning
        super(NoisyFastMRIDatasetBuilder, self).__init__(
            dataset=self.dataset,
            brain=self.brain,
            no_kspace=True,
            **kwargs,
        )
        if self.mode == 'test':
            raise NotImplementedError('Noisy dataset only works for train/val')

    def _preprocessing_train(self, _kspace, image, _contrast):
        image = scale_tensors(image, scale_factor=self.scale_factor)[0]
        image = image[..., None]
        noise_power = self.draw_noise_power()
        noise = tf.random.normal(
            shape=tf.shape(image),
            mean=0.0,
            stddev=noise_power,
            dtype=image.dtype,
        )
        image_noisy = image + noise
        model_inputs = (image_noisy,)
        if self.noise_input:
            model_inputs += (noise_power,)
        if self.residual_learning:
            model_outputs = noise
        else:
            model_outputs = image
        return model_inputs, model_outputs

    def draw_noise_power(self):
        noise_power = tf.random.uniform(
            (1,),
            minval=self.noise_power[0],
            maxval=self.noise_power[1],
        )
        return noise_power

    def preprocessing(self, *data_tensors):
        preprocessing_outputs = self._preprocessing_train(*data_tensors)
        return preprocessing_outputs
