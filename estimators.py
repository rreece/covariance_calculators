"""
covariance_calculators/estimators.py

TODO:

[x] Sample mean and covariance
[ ] Support pandas DataFrames
[ ] Rolling mean and covariance
[ ] Exponential moving mean and covariance
[ ] Online mean and covariance
[ ] Shrinkage estimators
"""


import numpy as np


def calc_sample_mean(data, frequency=None):
    _mean = np.mean(data, axis=1)
    if not frequency is None:
        _mean = _mean * frequency
    return _mean


def calc_sample_covariance(data, frequency=None):
    covariance = np.cov(data, rowvar=False, bias=False)  # bias=False uses n-1
    if not frequency is None:
        covariance = covariance * frequency
    return covariance

