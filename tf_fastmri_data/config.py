import os

FASTMRI_DATA_DIR = os.environ.get('FASTMRI_DATA_DIR', None)

SINGLECOIL_TRAIN_DIR = os.environ.get('SINGLECOIL_TRAIN_DIR', 'singlecoil_train/')
SINGLECOIL_VAL_DIR = os.environ.get('SINGLECOIL_VAL_DIR', 'singlecoil_val/')
SINGLECOIL_TEST_DIR = os.environ.get('SINGLECOIL_TEST_DIR', 'singlecoil_test/')

MULTICOIL_TRAIN_DIR = os.environ.get('MULTICOIL_TRAIN_DIR', 'multicoil_train/')
MULTICOIL_VAL_DIR = os.environ.get('MULTICOIL_VAL_DIR', 'multicoil_val/')
MULTICOIL_TEST_DIR = os.environ.get('MULTICOIL_TEST_DIR', 'multicoil_test/')

BRAIN_MULTICOIL_TRAIN_DIR = os.environ.get('BRAIN_MULTICOIL_TRAIN_DIR', 'brain_multicoil_train/')
BRAIN_MULTICOIL_VAL_DIR = os.environ.get('BRAIN_MULTICOIL_VAL_DIR', 'brain_multicoil_val/')
BRAIN_MULTICOIL_TEST_DIR = os.environ.get('BRAIN_MULTICOIL_TEST_DIR', 'brain_multicoil_test/')

PATHS_MAP = {
    # multicoil
    True: {
        # brain
        True: {
            'train': BRAIN_MULTICOIL_TRAIN_DIR,
            'val': BRAIN_MULTICOIL_VAL_DIR,
            'test': BRAIN_MULTICOIL_TEST_DIR,
        },
        # not brain
        False: {
            'train': MULTICOIL_TRAIN_DIR,
            'val': MULTICOIL_VAL_DIR,
            'test': MULTICOIL_TEST_DIR,
        },
    },
    # singlecoil
    False: {
        # not brain (brain is not available in single coil)
        False: {
            'train': SINGLECOIL_TRAIN_DIR,
            'val': SINGLECOIL_VAL_DIR,
            'test': SINGLECOIL_TEST_DIR,
        },
    },
}
