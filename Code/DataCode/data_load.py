import numpy as np
import random
from typing import Union, List
from Utils.utils import seed_everything, one_hot_encode


def uniform_generator(seed):
    seed_everything(seed)
    return lambda N: np.random.randint(0, N)


def gaussian_generator(seed):
    seed_everything(seed)
    return lambda N: min(max(int(np.random.normal(N / 2, N / 6)), 0), N - 1)


def poisson_generator(seed, lam=3):
    seed_everything(seed)
    return lambda N: min(np.random.poisson(lam), N - 1)


def exponential_generator(seed, scale=2):
    seed_everything(seed)
    return lambda N: min(int(np.random.exponential(scale)), N - 1)


def beta_generator(seed, a=2, b=5):
    seed_everything(seed)
    return lambda N: int(np.random.beta(a, b) * (N - 1))


def binomial_generator(seed, n=10, p=0.5):
    seed_everything(seed)
    return lambda N: min(np.random.binomial(n, p), N - 1)


def lognormal_generator(seed, mean=0, sigma=1):
    seed_everything(seed)
    return lambda N: min(int(np.random.lognormal(mean, sigma)), N - 1)


def mixed_generator(seed, distributions):
    seed_everything(seed)
    choices = [get_generator[dist](seed) for dist in distributions]
    return lambda N: random.choice(choices)(N)


get_generator = {
    "uniform": uniform_generator,
    "gaussian": gaussian_generator,
    "poisson": poisson_generator,
    "exponential": exponential_generator,
    "beta": beta_generator,
    "binomial": binomial_generator,
    "lognormal": lognormal_generator,
}


class RandomDataLoader:
    def __init__(self, N: int, dist: Union[str, List[str]] = "uniform", one_hot: bool = True, seed: int = 42):
        """
        Data Loader class that returns 2 random integers in the range [0, N-1]

        Parameters:
        N       : Upper limit of generated integer (exclusive)
        dist    : A distribution name (str) or a list of distributions to use (default: "uniform")
        one_hot : Whether to return one-hot encoded vectors or just integers
        seed    : Seed for random number generator
        """
        self.N = N
        self.one_hot = one_hot
        
        if isinstance(dist, str):
            dist = [dist]
        
        for d in dist:
            if d not in get_generator:
                raise ValueError(f"Unknown distribution: {d}")
        
        self.generator = mixed_generator(seed, dist) if len(dist) > 1 else get_generator[dist[0]](seed)

    def get_data(self):
        """Returns two randomly generated integers (or one-hot encoded vectors)"""
        x1 = self.generator(self.N)
        x2 = self.generator(self.N)
        if self.one_hot:
            return one_hot_encode(x1, self.N), one_hot_encode(x2, self.N)
        return x1, x2

    def __call__(self):
        return self.get_data()