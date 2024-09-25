import math

import fire
import numpy as np


def correct_variance(sample_variance, n):
    # https://en.wikipedia.org/wiki/Bessel%27s_correction
    return sample_variance * (n / (n - 1))


def correct_std(sample_std, n):
    return np.sqrt(correct_variance(sample_std**2, n))


def test_standard_deviation_correct(aimed_std, num_samples, experiments=100000):
    std = correct_std(aimed_std, num_samples)
    samples = np.random.normal(0, std, num_samples * experiments).reshape(
        experiments, num_samples
    )
    std_avg = np.sqrt(np.sum(np.var(samples, axis=1)) / experiments)
    print(aimed_std, std_avg, math.isclose(aimed_std, std_avg, rel_tol=1e-2))


if __name__ == "__main__":
    fire.Fire(test_standard_deviation_correct)
