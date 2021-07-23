import h5py
import numpy as np
import pytest
import tensorflow as tf
from tf_fastmri_data.test_utils import create_full_fastmri_tmp_dataset, create_data, \
    ktraj_function, K_shape_single_coil, K_shape_multi_coil, I_shape, contrast, fake_xml


@pytest.fixture(scope='session', autouse=False)
def ktraj():
    return ktraj_function

@pytest.fixture(scope="session", autouse=False)
def create_full_fastmri_test_tmp_dataset(tmpdir_factory):
    return create_full_fastmri_tmp_dataset(tmpdir_factory)
