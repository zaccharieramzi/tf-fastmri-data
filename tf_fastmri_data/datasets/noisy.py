import tensorflow as tf

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from tf_fastmri_data.preprocessing_utils.scaling import scale_tensors


class NoisyFastMRIDatasetBuilder(FastMRIDatasetBuilder):
    def __init__(
            self,
            scale_factor=1e6,
            noise_power_spec=30,
            noise_input=True,
            noise_mode='uniform',
            residual_learning=False,
            normal_noise_output=False,
            image_size=320,
            **kwargs,
        ):
        self.scale_factor = scale_factor
        self.noise_power_spec = noise_power_spec
        self.noise_input = noise_input
        self.noise_mode = noise_mode
        self.residual_learning = residual_learning
        self.normal_noise_output = normal_noise_output
        self.image_size = image_size
        super(NoisyFastMRIDatasetBuilder, self).__init__(
            no_kspace=True,
            **kwargs,
        )
        if self.mode == 'test':
            raise NotImplementedError('Noisy dataset only works for train/val')

    def _preprocessing_train(self, image):
        image = image[..., None]
        if self.image_size != 320:
            image = tf.image.resize(image, [self.image_size, self.image_size])
        image = scale_tensors(image, scale_factor=self.scale_factor)[0]
        noise_power = self.draw_noise_power(batch_size=tf.shape(image)[0])
        normal_noise = tf.random.normal(
            shape=tf.shape(image),
            mean=0.0,
            stddev=1.0,
            dtype=image.dtype,
        )
        noise_power_bdcast = noise_power[:, None, None, None]
        noise = normal_noise * noise_power_bdcast
        image_noisy = image + noise
        model_inputs = (image_noisy,)
        if self.noise_input:
            model_inputs += (noise_power,)
        if self.residual_learning:
            if self.normal_noise_output:
                model_outputs = normal_noise
            else:
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


class ComplexNoisyFastMRIDatasetBuilder(NoisyFastMRIDatasetBuilder):
    def __init__(self, **kwargs,):
        orig_prebuild = kwargs.get('prebuild', True)
        kwargs.update(dict(prebuild=False))
        super(ComplexNoisyFastMRIDatasetBuilder, self).__init__(
            complex_image=True,
            **kwargs,
        )
        self.no_kspace = False
        if orig_prebuild:
            self._build_datasets()

    def _preprocessing_train(self, image):
        image = scale_tensors(image, scale_factor=self.scale_factor)[0]
        image = image[..., None]
        noise_power = self.draw_noise_power(batch_size=tf.shape(image)[0])
        normal_noise = tf.random.normal(
            shape=tf.concat([tf.shape(image), [2]], axis=0),
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32,
        )
        normal_noise = tf.complex(normal_noise[..., 0], normal_noise[..., 1])
        noise_power_bdcast = noise_power[:, None, None, None]
        noise = normal_noise * tf.cast(noise_power_bdcast, normal_noise.dtype)
        image_noisy = image + noise
        model_inputs = (image_noisy,)
        if self.noise_input:
            model_inputs += (noise_power,)
        if self.residual_learning:
            if self.normal_noise_output:
                model_outputs = normal_noise
            else:
                model_outputs = noise
        else:
            model_outputs = image
        return model_inputs, model_outputs
