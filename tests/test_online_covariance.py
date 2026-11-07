"""
test_online_covariance.py
"""

import numpy as np
import pandas as pd
import pytest

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
    daily = EMACovariance(order=2, span=20, frequency=1)
    data = np.random.randn(100, 2) * 0.01  # daily returns

    for row in data:
        ema.add(row)
        daily.add(row)

    # Annualized mean and covariance should be exactly 252x daily
    assert np.allclose(ema.mean, daily.mean * 252, rtol=1e-12)
    assert np.allclose(ema.cov, daily.cov * 252, rtol=1e-12)


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
    raw_cov = np.cov(data[-252:], rowvar=False)
    assert np.allclose(sma.mean, raw_mean * 252, rtol=1e-6)
    assert np.allclose(sma.cov, raw_cov * 252, rtol=1e-6)


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


# --- Geometric mode tests ---


def test_geometric_online_matches_numpy_log():
    """Geometric OnlineCovariance should match numpy on log1p-transformed data."""
    np.random.seed(42)

    data = np.random.randn(200, 4) * 0.02  # realistic daily returns

    oc = OnlineCovariance(order=4, frequency=1, geometric=True)
    for row in data:
        oc.add(row)

    log_data = np.log1p(data)
    expected_mean = np.mean(log_data, axis=0)
    expected_cov = np.cov(log_data, rowvar=False)

    assert np.allclose(oc.mean, expected_mean, rtol=1e-6)
    assert np.allclose(oc.cov, expected_cov, rtol=1e-6)


def test_geometric_online_frequency_scaling():
    """Geometric mode with frequency should annualize log-return statistics."""
    np.random.seed(42)

    data = np.random.randn(100, 2) * 0.01
    oc = OnlineCovariance(order=2, frequency=252, geometric=True)

    for row in data:
        oc.add(row)

    log_data = np.log1p(data)
    raw_mean = np.mean(log_data, axis=0)
    raw_var = np.var(log_data, axis=0, ddof=1)

    assert np.allclose(oc.mean, raw_mean * 252, rtol=1e-6)
    assert np.allclose(np.diag(oc.cov), raw_var * 252, rtol=1e-6)


def test_geometric_mean_less_than_arithmetic():
    """Geometric mean should be less than arithmetic mean (Jensen's inequality).

    For positive r: log(1 + r) < r, so E[log(1 + r)] < E[r].
    """
    np.random.seed(42)

    # Use strictly positive returns
    data = np.abs(np.random.randn(200, 3)) * 0.03

    arith = OnlineCovariance(order=3, frequency=1, geometric=False)
    geo = OnlineCovariance(order=3, frequency=1, geometric=True)

    for row in data:
        arith.add(row)
        geo.add(row)

    assert np.all(geo.mean < arith.mean)


def test_geometric_ema_convergence():
    """Geometric EMA should converge to true log-return statistics for small alpha."""
    np.random.seed(42)

    n_samples = 30000
    n_features = 3
    true_mean = np.array([0.001, 0.002, -0.001])  # small daily returns
    true_cov = np.array([[0.0004, 0.0001, 0.00005],
                         [0.0001, 0.0009, -0.0002],
                         [0.00005, -0.0002, 0.0016]])
    data = np.random.multivariate_normal(mean=true_mean, cov=true_cov, size=n_samples)

    ema = EMACovariance(order=n_features, alpha=0.001, geometric=True)
    for row in data:
        ema.add(row)

    log_data = np.log1p(data)
    expected_mean = np.mean(log_data, axis=0)
    expected_cov = np.cov(log_data, rowvar=False)

    assert np.allclose(ema.mean, expected_mean, rtol=0, atol=1e-2)
    assert np.allclose(ema.cov, expected_cov, rtol=0, atol=2e-3)


def test_geometric_sma_matches_numpy_log():
    """Geometric SMA should match numpy on log1p of last N samples."""
    np.random.seed(42)

    window = 20
    data = np.random.randn(100, 2) * 0.02

    sma = SMACovariance(order=2, span=window, frequency=1, geometric=True)
    for row in data:
        sma.add(row)

    log_data = np.log1p(data[-window:])
    expected_mean = np.mean(log_data, axis=0)
    expected_cov = np.cov(log_data, rowvar=False)

    assert np.allclose(sma.mean, expected_mean, rtol=1e-6)
    assert np.allclose(sma.cov, expected_cov, rtol=1e-6)


def test_geometric_all_estimators_valid_output():
    """All estimators in geometric mode should produce valid covariance matrices."""
    np.random.seed(42)

    n_vars = 3
    estimators = [
        OnlineCovariance(order=n_vars, frequency=1, geometric=True),
        EMACovariance(order=n_vars, span=50, frequency=1, geometric=True),
        SMACovariance(order=n_vars, span=50, frequency=1, geometric=True),
    ]

    data = np.random.randn(100, n_vars) * 0.02

    for row in data:
        for est in estimators:
            est.add(row)

    for est in estimators:
        cov = est.cov
        assert cov.shape == (n_vars, n_vars)
        assert np.allclose(cov, cov.T, rtol=1e-6)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)


def test_geometric_false_is_default():
    """Geometric=False should be the default and produce identical results to before."""
    np.random.seed(42)

    data = np.random.randn(50, 3)

    oc_default = OnlineCovariance(order=3, frequency=1)
    oc_explicit = OnlineCovariance(order=3, frequency=1, geometric=False)

    for row in data:
        oc_default.add(row)
        oc_explicit.add(row)

    assert np.allclose(oc_default.mean, oc_explicit.mean, rtol=1e-12)
    assert np.allclose(oc_default.cov, oc_explicit.cov, rtol=1e-12)


def test_cagr_property():
    """cagr property should return exp(mean) - 1."""
    np.random.seed(42)

    data = np.random.randn(100, 3) * 0.02
    oc = OnlineCovariance(order=3, frequency=252, geometric=True)

    for row in data:
        oc.add(row)

    expected_cagr = np.expm1(oc.mean)
    assert np.allclose(oc.cagr, expected_cagr, rtol=1e-12)


def test_mean_return_geometric():
    """mean_return should return CAGR when geometric=True."""
    np.random.seed(42)

    data = np.random.randn(100, 3) * 0.02
    geo = OnlineCovariance(order=3, frequency=252, geometric=True)

    for row in data:
        geo.add(row)

    assert np.allclose(geo.mean_return, geo.cagr, rtol=1e-12)


def test_mean_return_arithmetic():
    """mean_return should return mean when geometric=False."""
    np.random.seed(42)

    data = np.random.randn(100, 3) * 0.02
    arith = OnlineCovariance(order=3, frequency=252, geometric=False)

    for row in data:
        arith.add(row)

    assert np.allclose(arith.mean_return, arith.mean, rtol=1e-12)


# --- EMA bias correction tests ---


def test_ema_matches_pandas_at_every_step():
    """Without warmup, both bias modes should match pandas from the second observation on."""
    np.random.seed(42)

    data = pd.DataFrame(np.random.randn(100, 2))
    alpha = 0.1

    for bias in [True, False]:
        ema = EMACovariance(order=2, alpha=alpha, warmup=0, bias=bias)
        ewm = data.ewm(alpha=alpha, adjust=False)
        expected_var = ewm.var(bias=bias).to_numpy()
        expected_cov = data[0].ewm(alpha=alpha, adjust=False).cov(data[1], bias=bias).to_numpy()

        for i, row in enumerate(data.to_numpy()):
            ema.add(row)
            if i == 0:
                assert ema.cov is None
                continue
            assert np.allclose(np.diag(ema.cov), expected_var[i], rtol=1e-9)
            assert np.isclose(ema.cov[0, 1], expected_cov[i], rtol=1e-9)


def test_ema_bias_correction_is_span_over_span_minus_one():
    """At steady state, the correction factor should be span / (span - 1)."""
    np.random.seed(42)

    span = 19
    biased = EMACovariance(order=2, span=span, bias=True)
    unbiased = EMACovariance(order=2, span=span)

    for row in np.random.randn(1000, 2):
        biased.add(row)
        unbiased.add(row)

    assert np.allclose(unbiased.cov, biased.cov * span / (span - 1), rtol=1e-9)
    assert np.allclose(unbiased.corr, biased.corr, rtol=1e-9)


def test_ema_unbiased_by_default():
    """Averaged over many streams, cov should equal the truth at every sample size."""
    rng = np.random.default_rng(42)

    n_streams = 4000
    true_cov = np.array([[1.0, 0.5],
                         [0.5, 2.0]])
    L = np.linalg.cholesky(true_cov)

    # span=50 has a warmup of 10 observations, warmup=0 has none
    for kwargs in [{"span": 50}, {"span": 50, "warmup": 0}]:
        for n_obs in [11, 30, 200]:
            total = np.zeros((2, 2))
            for _ in range(n_streams):
                ema = EMACovariance(order=2, **kwargs)
                for row in rng.standard_normal((n_obs, 2)) @ L.T:
                    ema.add(row)
                total += ema.cov
            assert np.allclose(total / n_streams, true_cov, rtol=0, atol=0.05)


def test_ema_bias_frequency_scaling():
    """Frequency scaling should be applied on top of the bias correction."""
    np.random.seed(42)

    plain = EMACovariance(order=2, span=20)
    annual = EMACovariance(order=2, span=20, frequency=252)

    for row in np.random.randn(100, 2) * 0.01:
        plain.add(row)
        annual.add(row)

    assert np.allclose(annual.cov, plain.cov * 252, rtol=1e-9)


def test_ema_checkpoint_preserves_bias_state():
    """A restored EMA should continue identically, including inside warmup."""
    np.random.seed(42)

    data = np.random.randn(80, 2)

    for bias in [True, False]:
        for cut in [3, 10, 50]:
            ema = EMACovariance(order=2, span=50, bias=bias)
            for row in data[:cut]:
                ema.add(row)
            restored = EMACovariance.from_dict(ema.to_dict())
            for row in data[cut:]:
                ema.add(row)
                restored.add(row)
            assert np.array_equal(ema.cov, restored.cov)


def test_ema_old_checkpoint_stays_uncorrected():
    """A checkpoint saved before the bias option should load in uncorrected mode."""
    np.random.seed(42)

    ema = EMACovariance(order=2, span=50, bias=True)
    for row in np.random.randn(80, 2):
        ema.add(row)

    state = ema.to_dict()
    del state["bias"]
    del state["sum_w2"]

    assert np.array_equal(EMACovariance.from_dict(state).cov, ema.cov)


# --- merge tests ---


def _split_and_merge(data, n_first, **kwargs):
    first = OnlineCovariance(data.shape[1], **kwargs)
    second = OnlineCovariance(data.shape[1], **kwargs)
    for row in data[:n_first]:
        first.add(row)
    for row in data[n_first:]:
        second.add(row)
    return first.merge(second)


@pytest.mark.parametrize("kwargs", [
    {"frequency": 252},
    {"geometric": True},
    {"frequency": 12, "geometric": True},
])
def test_merge_matches_single_pass_with_settings(kwargs):
    """A merged estimator should equal one fed every observation, and stay correct after more adds."""
    np.random.seed(42)

    data = np.random.randn(200, 3) * 0.01
    extra = np.random.randn(50, 3) * 0.01

    single = OnlineCovariance(3, **kwargs)
    for row in data:
        single.add(row)
    merged = _split_and_merge(data, 80, **kwargs)

    for name, value in kwargs.items():
        assert getattr(merged, name) == value
    assert np.allclose(merged.mean, single.mean, rtol=1e-9)
    assert np.allclose(merged.cov, single.cov, rtol=1e-9)
    assert np.allclose(merged.mean_return, single.mean_return, rtol=1e-9)

    for row in extra:
        merged.add(row)
        single.add(row)

    assert np.allclose(merged.mean, single.mean, rtol=1e-9)
    assert np.allclose(merged.cov, single.cov, rtol=1e-9)


def test_merge_preserves_dtype():
    first = OnlineCovariance(2, dtype=np.float32)
    second = OnlineCovariance(2, dtype=np.float32)
    assert first.merge(second).dtype == np.float32


def test_merge_with_empty():
    filled = OnlineCovariance(2, frequency=4)
    for row in [[1.0, 2.0], [2.0, 3.0], [4.0, 1.0]]:
        filled.add(row)
    empty = OnlineCovariance(2, frequency=4)

    for merged in [filled.merge(empty), empty.merge(filled)]:
        assert merged.count == filled.count
        assert np.allclose(merged.mean, filled.mean)
        assert np.allclose(merged.cov, filled.cov)

    both_empty = empty.merge(OnlineCovariance(2, frequency=4))
    assert both_empty.count == 0
    assert both_empty.mean is None


@pytest.mark.parametrize("other_kwargs", [
    {"frequency": 252},
    {"geometric": True},
    {"dtype": np.float32},
])
def test_merge_rejects_different_settings(other_kwargs):
    with pytest.raises(ValueError, match="different"):
        OnlineCovariance(2).merge(OnlineCovariance(2, **other_kwargs))


def test_merge_rejects_different_types():
    with pytest.raises(TypeError):
        OnlineCovariance(2).merge(SMACovariance(2, span=5))


@pytest.mark.parametrize("estimator", [
    EMACovariance(2, span=10),
    SMACovariance(2, span=10),
])
def test_merge_not_implemented_for_moving_estimators(estimator):
    with pytest.raises(NotImplementedError):
        estimator.merge(estimator)


# --- input handling tests ---


@pytest.mark.parametrize("make_estimator", [
    lambda: SMACovariance(2, span=10),
    lambda: EMACovariance(2, span=50),  # warmup of 10 covers the observations added below
])
def test_reused_buffer_does_not_corrupt_estimator(make_estimator):
    """Estimators that keep observations must not alias the caller's array."""
    np.random.seed(42)

    data = np.random.randn(30, 2)
    reused, fresh = make_estimator(), make_estimator()
    buffer = np.zeros(2)

    for row in data:
        buffer[:] = row
        reused.add(buffer)
        fresh.add(row.copy())

    assert np.array_equal(reused.cov, fresh.cov)


@pytest.mark.parametrize("observation", [
    (1.0, 2.0),
    [1.0, 2.0],
    np.array([1.0, 2.0]),
    np.array([[1.0, 2.0]]),
    pd.Series([1.0, 2.0]),
])
def test_add_accepts_array_like(observation):
    oc = OnlineCovariance(2)
    oc.add(observation)
    assert np.array_equal(oc.mean, [1.0, 2.0])


@pytest.mark.parametrize("observation", [1, 1.0, np.int64(1), np.float32(1.0)])
def test_add_accepts_scalars(observation):
    oc = OnlineCovariance(1)
    oc.add(observation)
    assert np.array_equal(oc.mean, [1.0])


def test_add_rejects_wrong_size():
    with pytest.raises(ValueError, match="size"):
        OnlineCovariance(2).add([1.0, 2.0, 3.0])


@pytest.mark.parametrize("make_estimator", [
    lambda: EMACovariance(2),
    lambda: EMACovariance(2, alpha=0),
    lambda: EMACovariance(2, alpha=1.5),
    lambda: EMACovariance(2, alpha=0.1, warmup=1),
    lambda: SMACovariance(2),
    lambda: SMACovariance(2, span=0),
])
def test_invalid_parameters_raise(make_estimator):
    with pytest.raises(ValueError):
        make_estimator()


def test_ema_corr_is_none_during_warmup():
    ema = EMACovariance(2, span=50)
    for i in range(3):
        ema.add([float(i), i * 2.0 + 1.0])

    assert ema.cov is None
    assert ema.corr is None


@pytest.mark.parametrize("make_estimator", [
    lambda: OnlineCovariance(1),
    lambda: SMACovariance(1, span=5),
    lambda: EMACovariance(1, span=10),
])
def test_single_feature_shapes(make_estimator):
    """cov and corr should have shape (1, 1) for every estimator."""
    np.random.seed(42)

    estimator = make_estimator()
    for value in np.random.randn(10):
        estimator.add(float(value))

    assert estimator.cov.shape == (1, 1)
    assert np.allclose(estimator.corr, [[1.0]])


# --- checkpoint tests ---


@pytest.mark.parametrize("make_estimator", [
    lambda: OnlineCovariance(2, frequency=4, geometric=True),
    lambda: SMACovariance(2, span=8, frequency=4),
])
@pytest.mark.parametrize("cut", [1, 5, 50])
def test_checkpoint_round_trip(make_estimator, cut):
    """A restored estimator should continue identically to the original."""
    np.random.seed(42)

    data = np.random.randn(80, 2) * 0.01
    estimator = make_estimator()
    for row in data[:cut]:
        estimator.add(row)

    restored = type(estimator).from_dict(estimator.to_dict())
    for row in data[cut:]:
        estimator.add(row)
        restored.add(row)

    assert np.array_equal(estimator.mean, restored.mean)
    assert np.array_equal(estimator.cov, restored.cov)


# --- geometric return validation tests ---


GEOMETRIC_ESTIMATORS = [
    lambda: OnlineCovariance(2, geometric=True),
    lambda: EMACovariance(2, span=10, geometric=True),
    lambda: SMACovariance(2, span=10, geometric=True),
]


@pytest.mark.parametrize("make_estimator", GEOMETRIC_ESTIMATORS)
@pytest.mark.parametrize("bad_return", [-1.0, -1.5])
def test_geometric_rejects_returns_at_or_below_minus_one(make_estimator, bad_return):
    """A bad return should raise and leave the state unchanged, so the stream can continue."""
    np.random.seed(42)

    data = np.random.randn(30, 2) * 0.01
    estimator, reference = make_estimator(), make_estimator()

    for i, row in enumerate(data):
        if i == 15:
            with pytest.raises(ValueError, match="greater than -1"):
                estimator.add([0.01, bad_return])
        estimator.add(row)
        reference.add(row)

    assert estimator.count == reference.count
    assert np.array_equal(estimator.mean, reference.mean)
    assert np.array_equal(estimator.cov, reference.cov)


@pytest.mark.parametrize("make_estimator", GEOMETRIC_ESTIMATORS)
def test_geometric_accepts_large_losses(make_estimator):
    """Losses just above -100% are valid and should stay finite."""
    estimator = make_estimator()
    for row in [[-0.999, 0.01], [0.02, -0.5], [0.01, 0.03], [0.0, -0.9]] * 5:
        estimator.add(row)

    assert np.all(np.isfinite(estimator.cov))


def test_non_geometric_allows_returns_below_minus_one():
    """The limit only applies in geometric mode: plain data can have any value."""
    oc = OnlineCovariance(1)
    oc.add(-5.0)
    assert oc.mean[0] == -5.0
