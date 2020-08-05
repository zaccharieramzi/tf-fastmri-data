# tf-fastmri-data
[![Build Status](https://travis-ci.com/zaccharieramzi/tf-fastmri-data.svg?branch=master)](https://travis-ci.com/zaccharieramzi/tf-fastmri-data)

Built around the `tf.data` API, `tf-fastmri-data` offers reliable, unit-tested, datasets for the fastMRI dataset.

## Installation

Currently, you need to install the package from source:

```
git clone https://github.com/zaccharieramzi/tf-fastmri-data.git
cd tf-fastmri-data
pip install .
```

## Example use

```
from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder

train_dataset = CartesianFastMRIDatasetBuilder(path='/path/to/singlecoil_train').preprocessed_ds
```

## Data

To download the data, you need to consent to the fastMRI terms listed [here](https://fastmri.med.nyu.edu/).
Afterwards, you should receive an email with data download links.

You can then use the environment variable `FASTMRI_DATA_DIR` to indicate where your fastMRI is.
This will allow you to not have to specify the path when instantiating a `FastMRIDatasetBuilder`.

## Citation

If you use the fastMRI data or this code in your research, please consider citing the fastMRI dataset paper:

```
@inproceedings{zbontar2018fastMRI,
  title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
  author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Matthew J. Muckley and Mary Bruno and Aaron Defazio and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and James Pinkerton and Duo Wang and Nafissa Yakubova and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1811.08839},
  year={2018}
}
```
