import numpy as np
import pytest

from tf_fastmri_data.datasets.noisy import NoisyFastMRIDatasetBuilder, ComplexNoisyFastMRIDatasetBuilder


kspace_shape = [2, 640, 322, 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('contrast', [None, file_contrast])
@pytest.mark.parametrize('slice_random', [True, False])
@pytest.mark.parametrize('noise_input', [True, False])
@pytest.mark.parametrize('noise_power', [30, (0, 50)])
@pytest.mark.parametrize('noise_mode', ['uniform', 'gaussian'])
@pytest.mark.parametrize('batch_size', [None, 2])
def test_noisy_dataset_train(create_full_fastmri_test_tmp_dataset, contrast, slice_random, noise_input, noise_power, noise_mode, batch_size):
    if not (noise_mode == 'gaussian' and isinstance(noise_power, tuple)) and not (batch_size and not slice_random):
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
        ds = NoisyFastMRIDatasetBuilder(
            path=path,
            contrast=contrast,
            slice_random=slice_random,
            noise_input=noise_input,
            noise_power_spec=noise_power,
            noise_mode=noise_mode,
            batch_size=batch_size,
        )
        if noise_input:
            (image_noisy, *_others), model_outputs = next(ds.preprocessed_ds.as_numpy_iterator())
        else:
            image_noisy, model_outputs = next(ds.preprocessed_ds.as_numpy_iterator())
        np.testing.assert_equal(model_outputs.shape[-3:], [320, 320, 1])
        np.testing.assert_equal(image_noisy.shape[-3:], [320, 320, 1])
        np.testing.assert_equal(image_noisy.ndim, 4)
        np.testing.assert_equal(model_outputs.ndim, 4)

@pytest.mark.parametrize('contrast', [None, file_contrast])
@pytest.mark.parametrize('slice_random', [True, False])
@pytest.mark.parametrize('noise_input', [True, False])
@pytest.mark.parametrize('noise_power', [30, (0, 50)])
@pytest.mark.parametrize('noise_mode', ['uniform', 'gaussian'])
@pytest.mark.parametrize('batch_size', [None, 2])
def test_complex_noisy_dataset_train(create_full_fastmri_test_tmp_dataset, contrast, slice_random, noise_input, noise_power, noise_mode, batch_size):
    if not (noise_mode == 'gaussian' and isinstance(noise_power, tuple)) and not (batch_size and not slice_random):
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
        ds = ComplexNoisyFastMRIDatasetBuilder(
            path=path,
            contrast=contrast,
            slice_random=slice_random,
            noise_input=noise_input,
            noise_power_spec=noise_power,
            noise_mode=noise_mode,
            batch_size=batch_size,
        )
        if noise_input:
            (image_noisy, *_others), model_outputs = next(ds.preprocessed_ds.as_numpy_iterator())
        else:
            image_noisy, model_outputs = next(ds.preprocessed_ds.as_numpy_iterator())
        if not (batch_size == 2 and slice_random):
            # NOTE: for now complex images can only be of size 320 x 320
            np.testing.assert_equal(model_outputs.shape[-3:], (320, 320, 1))
            np.testing.assert_equal(image_noisy.shape[-3:], (320, 320, 1))
        else:
            assert model_outputs.shape[-2] == 320
            assert image_noisy.shape[-2] == 320
        np.testing.assert_equal(image_noisy.ndim, 4)
        assert image_noisy.dtype == np.complex64
        np.testing.assert_equal(model_outputs.ndim, 4)
