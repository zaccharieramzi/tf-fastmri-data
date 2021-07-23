import pytest

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.test_utils import key_map


@pytest.mark.parametrize('dataset', ['train', 'test', 'val'])
@pytest.mark.parametrize('multicoil', [True, False])
def test_dataset_builder_init(create_full_fastmri_test_tmp_dataset, dataset, multicoil):
    path = create_full_fastmri_test_tmp_dataset[key_map[multicoil][dataset]]
    FastMRIDatasetBuilder(path=path, dataset=dataset, multicoil=multicoil, prebuild=False)

