from contextlib import ExitStack

import numpy as np
import pytest
import tensorflow_io as tfio

from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder


kspace_shape = [2, 640, 322, 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('mask_mode', ['equidistant', 'random'])
@pytest.mark.parametrize('output_shape_spec', [True, False])
@pytest.mark.parametrize('multicoil', [True, False])
@pytest.mark.parametrize('contrast', [None, file_contrast])
@pytest.mark.parametrize('slice_random', [True, False])
def test_cartesian_dataset_train(create_full_fastmri_test_tmp_dataset, mask_mode, output_shape_spec, multicoil, contrast, slice_random):
    if multicoil:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    else:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = CartesianFastMRIDatasetBuilder(
        path=path,
        mask_mode=mask_mode,
        brain=output_shape_spec,
        output_shape_spec=output_shape_spec,
        multicoil=multicoil,
        contrast=contrast,
        slice_random=slice_random,
    )
    (kspace, mask, *_others), image = next(ds.preprocessed_ds.as_numpy_iterator())
    np.testing.assert_equal(kspace.shape[-3:], kspace_shape[1:])
    np.testing.assert_equal(mask.shape[-2:], [1, kspace_shape[-2]])
    np.testing.assert_equal(image.shape[-3:], [320, 320, 1])
    np.testing.assert_equal(image.ndim, 4)

@pytest.mark.parametrize('multicoil', [True, False])
def test_cartesian_dataset_train_batched(create_full_fastmri_test_tmp_dataset, multicoil):
    if multicoil:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    else:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    target_image_size = (640, 400)
    batch_size = 2
    ds = CartesianFastMRIDatasetBuilder(
        path=path,
        multicoil=multicoil,
        slice_random=True,
        batch_size=batch_size,
        target_image_size=target_image_size,
    )
    (kspace, mask, *_others), image = next(ds.preprocessed_ds.as_numpy_iterator())
    np.testing.assert_equal(kspace.shape[-3:], tuple(*target_image_size, 1))
    np.testing.assert_equal(mask.shape[-2:], [batch_size, target_image_size[1]])
    np.testing.assert_equal(image.shape[-3:], [320, 320, 1])
    np.testing.assert_equal(image.ndim, 4)
    np.testing.assert_equal(kspace.shape[0], batch_size)

@pytest.mark.parametrize('mask_mode', ['equidistant', 'random'])
@pytest.mark.parametrize('output_shape_spec', [True, False])
@pytest.mark.parametrize('multicoil', [True, False])
@pytest.mark.parametrize('contrast', [None, file_contrast])
def test_cartesian_dataset_test(create_full_fastmri_test_tmp_dataset, mask_mode, output_shape_spec, multicoil, contrast):
    if multicoil:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_test']
    else:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_test']
    tfio_version_flag = tuple(int(v) for v in tfio.__version__.split('.')) <= (0, 15, 0)
    with ExitStack() as stack:
        if tfio_version_flag:
            stack.enter_context(pytest.raises(ValueError))
        ds = CartesianFastMRIDatasetBuilder(
            dataset='test',
            path=path,
            mask_mode=mask_mode,
            output_shape_spec=output_shape_spec,
            brain=output_shape_spec,
            multicoil=multicoil,
            contrast=contrast,
        )
    if not tfio_version_flag:
        kspace, mask, *_others = next(ds.preprocessed_ds.as_numpy_iterator())
        np.testing.assert_equal(kspace.shape[-3:], kspace_shape[1:])
        np.testing.assert_equal(mask.shape[-2:], [1, kspace_shape[-2]])
