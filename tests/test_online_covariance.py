"""
test_online_covariance.py
"""

import math
import numpy as np

from covariance_calculators.estimators import calc_sample_mean, calc_sample_covariance, OnlineCovariance, EMACovariance


np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def test_online_covariance():

    # Generate sample data
    n_samples = 10000
    n_features = 3
    true_mean = np.array([1, 2, 3])
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.004],
                         [0.003, -0.004,  0.040]])
    data = np.random.multivariate_normal(mean=true_mean,
                                   cov=true_cov,
                                   size=n_samples)

    # Calculate sample mean and covariance
    mean = calc_sample_mean(data)
    covariance = calc_sample_covariance(data)

    # Check that sample statistics are close to the truth
    assert np.allclose(mean, true_mean, rtol=0, atol=1e-2)
    assert np.allclose(covariance, true_cov, rtol=0, atol=1e-3)

    # Calculate online mean and covariance
    online_calculator = OnlineCovariance(n_features)

    for row in data:
        online_calculator.add(row)

    # Check that online statistics are virtually identical to the sample statistics
    assert np.allclose(online_calculator.mean, mean, rtol=0, atol=1e-5)
    assert np.allclose(online_calculator.cov, covariance, rtol=0, atol=1e-5)


def test_ema_covariance():

    # Generate sample data
    n_samples = 10000
    n_features = 3
    true_mean = np.array([1, 2, 3])
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.004],
                         [0.003, -0.004,  0.040]])
    data = np.random.multivariate_normal(mean=true_mean,
                                   cov=true_cov,
                                   size=n_samples)
    # Calculate EMA mean and covariance
    ema_calculator = EMACovariance(n_features, alpha=0.001)

    for row in data:
        ema_calculator.add(row)

    # Check that EMA statistics are nearly identical to the truth for small alpha
    assert np.allclose(ema_calculator.mean, true_mean, rtol=0, atol=1e-2)
    assert np.allclose(ema_calculator.cov, true_cov, rtol=0, atol=3e-3)


def test_online_covariance_merge():

    # Generate sample data with two differently correllated datasets
    n_samples = 10000
    n_features = 3
    true_mean_1 = np.array([2.2, 4.4, 1.5])
    true_mean_2 = np.array([5, 6, 2])
    true_cov = np.array([[0.015,  0.008,  0.003],
                         [0.008,  0.057, -0.021],
                         [0.003, -0.021,  0.040]])
    data_1 = np.random.multivariate_normal(
                        mean=true_mean_1,
                        cov=true_cov,
                        size=int(n_samples/2))
    data_2 = np.random.multivariate_normal(
                        mean=true_mean_2,
                        cov=true_cov,
                        size=n_samples)

    ocov_calc_1 = OnlineCovariance(n_features)
    ocov_calc_2 = OnlineCovariance(n_features)
    ocov_calc_both = OnlineCovariance(n_features)
    
    # Save online-covariances for part 1 and 2 separately but also
    # put all observations into the OnlineCovariance object for both.
    
    for row in data_1:
        ocov_calc_1.add(row)
        ocov_calc_both.add(row)
        
    for row in data_2:
        ocov_calc_2.add(row)
        ocov_calc_both.add(row)
        
    ocov_calc_merged = ocov_calc_1.merge(ocov_calc_2)
    
    assert ocov_calc_both.count == ocov_calc_merged.count, \
        """
        Count of both and merged should be the same.
        """
    assert np.allclose(ocov_calc_both.mean, ocov_calc_merged.mean), \
        """
        Mean of both and merged should be the same.
        """
    assert np.allclose(ocov_calc_both.cov, ocov_calc_merged.cov), \
        """
        Covarance-matrix of both and merged should be the same.
        """
    assert np.allclose(ocov_calc_both.corr, ocov_calc_merged.corr), \
        """
        Pearson-Correlationcoefficient-matrix of both and merged should be the same.
        """

