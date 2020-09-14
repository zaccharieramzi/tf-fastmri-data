import numpy as np
import pytest

from tf_fastmri_data.datasets.noisy import NoisyFastMRIDatasetBuilder


kspace_shape = [2, 640, 322, 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('contrast', [None, file_contrast])
@pytest.mark.parametrize('slice_random', [True, False])
@pytest.mark.parametrize('noise_input', [True, False])
@pytest.mark.parametrize('noise_power', [30, (0, 50)])
def test_cartesian_dataset_train(create_full_fastmri_test_tmp_dataset, contrast, slice_random, noise_input, noise_power):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = NoisyFastMRIDatasetBuilder(
        path=path,
        contrast=contrast,
        slice_random=slice_random,
    )
    (image_noisy, *_others), image = next(ds.preprocessed_ds.as_numpy_iterator())
    np.testing.assert_equal(image.shape[-3:], [320, 320, 1])
    np.testing.assert_equal(image_noisy.shape[-3:], [320, 320, 1])
