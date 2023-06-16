import os
import random
import numpy as np
import tensorflow as tf


def enable_reproducibility(random_seed):
    """ Source: https://github.com/NVIDIA/framework-determinism """

    if random_seed is not None:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)