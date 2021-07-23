import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# taken from https://github.com/CEA-COSMIC/ModOpt/blob/master/setup.py
with open('requirements.txt') as open_file:
    install_requires = open_file.read()

import tf_fastmri_data

setuptools.setup(
    name="tf-fastmri-data",
    version=tf_fastmri_data.__version__,
    author=tf_fastmri_data.__author__,
    author_email=tf_fastmri_data.__author_email__,
    description=tf_fastmri_data.__docs__,
    license=tf_fastmri_data.__license__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=tf_fastmri_data.__homepage__,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
)
