import pytest

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder


@pytest.mark.parametrize('dataset', ['train', 'test', 'val'])
@pytest.mark.parametrize('multicoil', [True, False])
def test_dataset_builder_init(dataset, multicoil):
    FastMRIDatasetBuilder(dataset=dataset, multicoil=multicoil, prebuild=False)

def test_dataset_builder_error():
    with pytest.raises(NotImplementedError):
        FastMRIDatasetBuilder(prebuild=True)
