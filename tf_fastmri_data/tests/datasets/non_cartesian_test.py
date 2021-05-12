import numpy as np
import pytest

from tf_fastmri_data.datasets.non_cartesian import NonCartesianFastMRIDatasetBuilder



@pytest.mark.parametrize('acq_type', ['radial', 'cartesian', 'spiral'])
@pytest.mark.parametrize('dcomp', [True, False])
@pytest.mark.parametrize('multicoil', [True, False])
@pytest.mark.parametrize('slice_random', [True, False])
@pytest.mark.parametrize('crop_image_data', [True, False])
@pytest.mark.parametrize('image_size', [(20, 20)])
def test_non_cartesian_dataset_train(create_full_fastmri_test_tmp_dataset, acq_type, dcomp, multicoil, slice_random, crop_image_data, image_size):
    if multicoil:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    else:
        path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = NonCartesianFastMRIDatasetBuilder(
        path=path,
        acq_type=acq_type,
        dcomp=dcomp,
        crop_image_data=crop_image_data,
        multicoil=multicoil,
        slice_random=slice_random,
        image_size=image_size,
    )
    (kspace, traj, *_others), image = next(ds.preprocessed_ds.as_numpy_iterator())
    # CHeck that the kspace and trajectory match in shape!
    np.testing.assert_equal(kspace.shape[-2], traj.shape[-1])
    if multicoil:
        if crop_image_data:
            smaps, output_shape, op_args = _others
            np.testing.assert_equal(output_shape[0], image.shape[-3:])
        else:
            smaps, op_args = _others
        # Check the size of smaps
        np.testing.assert_equal(smaps.shape[-3:-2], kspace.shape[-3:-2])
        np.testing.assert_equal(smaps.shape[-2:], image_size)
    else:
        if crop_image_data:
            output_shape, op_args = _others
            np.testing.assert_equal(output_shape[0], image.shape[-3:])
        else:
            op_args = _others
    if dcomp:
        orig_shape, dcomp = op_args
        np.testing.assert_equal(dcomp.shape[-1], traj.shape[-1])
    else:
        orig_shape = op_args
    np.testing.assert_equal(orig_shape[0], image_size[-1])
    np.testing.assert_equal(image.shape[-3:], [320, 320, 1])
    np.testing.assert_equal(image.ndim, 4)
