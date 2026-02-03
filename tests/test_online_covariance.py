"""
test_online_covariance.py
"""

import numpy as np

from covariance_calculators.estimators import calc_sample_mean, calc_sample_covariance, OnlineCovariance, EMACovariance, SMACovariance


np.set_printoptions(precision=4, suppress=True)


def test_online_covariance():
    np.random.seed(42)

    # Generate sample data
    n_samples = 30000
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
    assert np.allclose(online_calculator.mean, mean, rtol=1e-6, atol=0)
    assert np.allclose(online_calculator.cov, covariance, rtol=1e-6, atol=0)


def test_online_covariance_basic_mean_and_variance():
    """Test that mean and variance are computed correctly for simple known data."""
    oc = OnlineCovariance(order=2, frequency=1)

    # Add known data points
    data = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
    ])

    for row in data:
        oc.add(row)

    # Expected mean
    expected_mean = np.array([3.0, 6.0])
    assert np.allclose(oc.mean, expected_mean, rtol=1e-6)

    # Expected variance (sample variance with n-1 denominator)
    expected_var = np.array([2.5, 10.0])
    assert np.allclose(np.diag(oc.cov), expected_var, rtol=1e-6)


def test_online_covariance_matches_numpy():
    """Test that OnlineCovariance matches numpy computation exactly."""
    np.random.seed(123)

    oc = OnlineCovariance(order=4, frequency=1)
    data = np.random.randn(200, 4)

    for row in data:
        oc.add(row)

    expected_mean = np.mean(data, axis=0)
    expected_cov = np.cov(data, rowvar=False)

    assert np.allclose(oc.mean, expected_mean, rtol=1e-6)
    assert np.allclose(oc.cov, expected_cov, rtol=1e-6)


def test_ema_covariance():
    np.random.seed(42)

    # Generate sample data
    n_samples = 30000
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
    assert np.allclose(ema_calculator.cov, true_cov, rtol=0, atol=2e-3)


def test_online_covariance_merge():
    np.random.seed(42)

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


def test_online_covariance_single_observation():
    """Test behavior with only one observation."""
    oc = OnlineCovariance(order=2, frequency=1)
    oc.add(np.array([1.0, 2.0]))

    # Mean should be the single observation
    assert np.allclose(oc.mean, np.array([1.0, 2.0]))

    # Covariance is undefined with n=1, returns None
    assert oc.cov is None


def test_online_covariance_perfect_correlation():
    """Test covariance with perfectly correlated data."""
    oc = OnlineCovariance(order=2, frequency=1)

    # y = 2x, perfect positive correlation
    for x in range(1, 11):
        oc.add(np.array([float(x), 2.0 * x]))

    corr = oc.corr
    assert np.allclose(corr[0, 1], 1.0, rtol=1e-6)
    assert np.allclose(corr[1, 0], 1.0, rtol=1e-6)


def test_online_covariance_negative_correlation():
    """Test covariance with perfectly negatively correlated data."""
    oc = OnlineCovariance(order=2, frequency=1)

    # y = -2x, perfect negative correlation
    for x in range(1, 11):
        oc.add(np.array([float(x), -2.0 * x]))

    corr = oc.corr
    assert np.allclose(corr[0, 1], -1.0, rtol=1e-6)


def test_online_covariance_frequency_scaling():
    """Test that frequency scaling is applied correctly."""
    oc = OnlineCovariance(order=2, frequency=252)  # annualize daily returns

    data = np.array([
        [0.01, 0.02],
        [0.02, 0.03],
        [-0.01, -0.01],
        [0.005, 0.01],
    ])

    for row in data:
        oc.add(row)

    # Mean should be scaled by frequency
    raw_mean = np.mean(data, axis=0)
    assert np.allclose(oc.mean, raw_mean * 252, rtol=1e-6)

    # Variance should be scaled by frequency
    raw_var = np.var(data, axis=0, ddof=1)
    assert np.allclose(np.diag(oc.cov), raw_var * 252, rtol=1e-6)


def test_ema_alpha_halflife_span_consistency():
    """Test that alpha, halflife, and span are consistent."""
    np.random.seed(42)

    # span=19 gives alpha = 2/20 = 0.1
    # Use warmup=0 for all to test alpha consistency without warmup effects
    ema_span = EMACovariance(order=2, span=19, warmup=0, frequency=1)

    # From code: alpha = 1 - exp(-ln(2)/halflife)
    # So: halflife = -ln(2) / ln(1-alpha)
    # For alpha=0.1, halflife ~= 6.58
    halflife = -np.log(2) / np.log(1 - 0.1)
    ema_halflife = EMACovariance(order=2, halflife=halflife, warmup=0, frequency=1)

    # Direct alpha
    ema_alpha = EMACovariance(order=2, alpha=0.1, warmup=0, frequency=1)

    data = np.random.randn(50, 2)

    for row in data:
        ema_span.add(row)
        ema_halflife.add(row)
        ema_alpha.add(row)

    # All should produce same results
    assert np.allclose(ema_span.cov, ema_alpha.cov, rtol=1e-6)
    assert np.allclose(ema_halflife.cov, ema_alpha.cov, rtol=1e-6)


def test_ema_frequency_scaling():
    """Test that frequency scaling works for EMA."""
    np.random.seed(42)

    ema = EMACovariance(order=2, span=20, frequency=252)
    data = np.random.randn(100, 2) * 0.01  # daily returns

    for row in data:
        ema.add(row)

    # Annualized mean and variance should be ~252x daily
    assert ema.mean.shape == (2,)
    assert ema.cov.shape == (2, 2)


def test_sma_covariance():
    """Test basic SMA covariance with window."""
    np.random.seed(42)

    window = 20
    sma = SMACovariance(order=2, span=window, frequency=1)
    data = np.random.randn(100, 2)

    for row in data:
        sma.add(row)

    # After 100 observations, should use only last 20
    expected_mean = np.mean(data[-window:], axis=0)
    expected_cov = np.cov(data[-window:], rowvar=False)

    assert np.allclose(sma.mean, expected_mean, rtol=1e-6)
    assert np.allclose(sma.cov, expected_cov, rtol=1e-6)


def test_sma_window_filling():
    """Test SMA behavior while window is filling."""
    window = 10
    sma = SMACovariance(order=2, span=window, frequency=1)

    data = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    for row in data:
        sma.add(row)

    # With only 3 observations, should use all 3
    expected_mean = np.mean(data, axis=0)
    assert np.allclose(sma.mean, expected_mean, rtol=1e-6)


def test_sma_window_sliding():
    """Test that old observations are dropped correctly."""
    window = 5
    sma = SMACovariance(order=2, span=window, frequency=1)

    # Add 10 observations
    for i in range(10):
        sma.add(np.array([float(i), float(i * 2)]))

    # Mean should be based on last 5: [5,6,7,8,9]
    expected_mean = np.array([7.0, 14.0])
    assert np.allclose(sma.mean, expected_mean, rtol=1e-6)


def test_sma_frequency_scaling():
    """Test frequency scaling for SMA."""
    np.random.seed(42)

    sma = SMACovariance(order=2, span=252, frequency=252)
    data = np.random.randn(300, 2) * 0.01

    for row in data:
        sma.add(row)

    # Should produce annualized values
    raw_mean = np.mean(data[-252:], axis=0)
    assert np.allclose(sma.mean, raw_mean * 252, rtol=1e-6)


def test_all_estimators_valid_output():
    """Test that all estimators produce valid covariance matrices."""
    np.random.seed(42)

    n_vars = 3
    estimators = [
        OnlineCovariance(order=n_vars, frequency=1),
        EMACovariance(order=n_vars, span=50, frequency=1),
        SMACovariance(order=n_vars, span=50, frequency=1),
    ]

    data = np.random.randn(100, n_vars)

    for row in data:
        for est in estimators:
            est.add(row)

    for est in estimators:
        cov = est.cov
        # Valid shape
        assert cov.shape == (n_vars, n_vars)
        # Symmetric
        assert np.allclose(cov, cov.T, rtol=1e-6)
        # Positive semidefinite (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

