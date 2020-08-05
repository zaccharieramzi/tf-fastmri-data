import pytest

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
        output_shape_spec=output_shape_spec,
        multicoil=multicoil,
        contrast=contrast,
        slice_random=slice_random,
    )
    (kspace, mask, *_others), image = next(ds.as_numpy_iterator())
    assert kspace.shape[-3:] == kspace_shape[1:]
    assert mask.shape[-3:] == [1, 1, kspace_shape[-1]]
    assert image.shape[-3:] == [320, 320, 1]
