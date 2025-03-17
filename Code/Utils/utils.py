import random
import os
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def one_hot_encode(value: int, N: int) -> np.ndarray:
    one_hot = np.zeros(N, dtype=int)
    one_hot[value] = 1
    return one_hot