import pytest

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder


key_map = {
    # multicoil
    True: {
        'train': 'fastmri_tmp_multicoil_train',
        'val': 'fastmri_tmp_multicoil_val',
        'test': 'fastmri_tmp_multicoil_test',
    },
    # not multicoil
    False: {
        'train': 'fastmri_tmp_singlecoil_train',
        'val': 'fastmri_tmp_singlecoil_val',
        'test': 'fastmri_tmp_singlecoil_test',
    },
}

@pytest.mark.parametrize('dataset', ['train', 'test', 'val'])
@pytest.mark.parametrize('multicoil', [True, False])
def test_dataset_builder_init(create_full_fastmri_test_tmp_dataset, dataset, multicoil):
    path = create_full_fastmri_test_tmp_dataset[key_map[multicoil][dataset]]
    FastMRIDatasetBuilder(path=path, dataset=dataset, multicoil=multicoil, prebuild=False)

