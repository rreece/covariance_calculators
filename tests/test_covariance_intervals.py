"""
test_covariance_intervals.py
"""

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

from covariance_calculators.estimators import calc_sample_covariance
from covariance_calculators.intervals import calc_covariance_intervals, calc_precision_intervals


mpl.use("Agg")
np.set_printoptions(precision=4, suppress=True)

IS_MACOS = sys.platform == "darwin"


def test_normal_covariance_interval():
    # Use default_rng instead of np.random.seed for cross-platform reproducibility.
    # The legacy RandomState API can produce different results across platforms/NumPy versions.
    # np.random.seed(42)
    rng = np.random.default_rng(42)

    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = rng.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)

    # Calculate confidence interval
    confidence_level = 0.95
    method = "normal"
    covariance, covariance_lower, covariance_upper = calc_covariance_intervals(data=data, confidence_level=confidence_level, method=method)

    if IS_MACOS:
        ref_covariance = np.array([[ 0.0095,  0.0044,  0.0026],
                                   [ 0.0044,  0.0205, -0.0019],
                                   [ 0.0026, -0.0019,  0.031 ]])

    else:
        ref_covariance = np.array([[ 0.0096,  0.0043,  0.0035],
                                   [ 0.0043,  0.0206, -0.0021],
                                   [ 0.0035, -0.0021,  0.0307]])

    assert np.allclose(covariance, ref_covariance, rtol=0, atol=1e-4)
    assert np.allclose(covariance, true_cov, rtol=0, atol=3e-3)


def test_compare_methods():
    # Use default_rng instead of np.random.seed for cross-platform reproducibility.
    # The legacy RandomState API can produce different results across platforms/NumPy versions.
    # np.random.seed(42)
    rng = np.random.default_rng(42)

    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = rng.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)

    # Compare all methods
    confidence_level = 0.95
    compare_methods(data, confidence_level=confidence_level)


def compare_methods(
    data: np.ndarray,
    confidence_level: float = 0.95,
):
    """
    Compare normal approximation, Wishart, and bootstrap methods.
    """
    methods = ["normal", "wishart", "bootstrap"]

    # calculate the covariance once for all methods
    covariance1 = calc_sample_covariance(data)

    results = dict()
    for method in methods:
        results[method] = calc_covariance_intervals(data=data, covariance=covariance1, confidence_level=confidence_level, method=method)

    # Print results
    for method in methods:
        print(f"\n{method} method estimate:")
        covariance, covariance_lower, covariance_upper = results[method]
        print(covariance)
        print("lower:")
        print(covariance_lower)
        print("upper:")
        print(covariance_upper)

    assert np.allclose(results["normal"][0], results["wishart"][0], rtol=0, atol=1e-4)
    assert np.allclose(results["normal"][0], results["bootstrap"][0], rtol=0, atol=1e-4)

    return results


def generate_toy_datasets(n_toys, n_samples, true_cov, rng):
    """Pre-generate toy datasets for coverage testing."""
    n_features = true_cov.shape[0]
    datasets = []
    for _ in range(n_toys):
        data = rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=true_cov,
            size=n_samples
        )
        datasets.append(data)
    return datasets


def test_coverage():
    # Use default_rng instead of np.random.seed for cross-platform reproducibility.
    # The legacy RandomState API can produce different results across platforms/NumPy versions.
    # np.random.seed(42)
    rng = np.random.default_rng(42)
#    import hepplot as hep

    # F(z) = Phi(z) = (1/2) * (1 + erf(z/sqrt(2)))
    # Phi(z) = (1 - alpha/2)   For two-sided
    # alpha = 2*(1 - Phi(z))
    #       = 2*(1 - 0.5*(1+math.erf(z/math.sqrt(2))))
    def _z_to_alpha(z):
        return 2*(1 - 0.5*(1+math.erf(z/math.sqrt(2))))

    alphas = [ _z_to_alpha(_z) for _z in [1, 2, 3, 4] ]
    confidence_levels = [ 1.0 - _a for _a in alphas ]
    print(alphas)
    print(confidence_levels)

    n_samples = 1000
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])

    # Pre-generate datasets for normal method (more toys)
    n_toys_normal = 1000
    datasets_normal = generate_toy_datasets(n_toys_normal, n_samples, true_cov, rng)

    # Pre-generate datasets for wishart method (fewer toys for speed)
    # Use a subset of the normal datasets for fair comparison
    n_toys_wishart = 100
    datasets_wishart = datasets_normal[:n_toys_wishart]

    # normal method experiments
    normal_coverages = list()
    for cl in confidence_levels:
        coverage = run_coverage_test(confidence_level=cl, method="normal", datasets=datasets_normal, true_cov=true_cov)
        avg_coverage = np.average(coverage)
        normal_coverages.append(avg_coverage)

    normal_coverage_alphas = [ 1.0 - _c for _c in normal_coverages ]
    print(normal_coverage_alphas)

    # wishart method experiments (uses same data as normal, just fewer samples)
    wishart_coverages = list()
    for cl in confidence_levels:
        coverage = run_coverage_test(confidence_level=cl, method="wishart", datasets=datasets_wishart, true_cov=true_cov)
        avg_coverage = np.average(coverage)
        wishart_coverages.append(avg_coverage)

    wishart_coverage_alphas = [ 1.0 - _c for _c in wishart_coverages ]
    print(wishart_coverage_alphas)

    # make coverage plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = 1 - p_\mathrm{CL}$")
    ax.set_ylabel(r"$\alpha_\mathrm{coverage} = 1 - p_\mathrm{coverage}$")
    ax.plot(alphas, alphas, color="darkgray", label="Perfect calibration")
    ax.plot(alphas, normal_coverage_alphas, marker='o', color="#1f77b4", label="Asymptotic interval")
    ax.plot(alphas, wishart_coverage_alphas, marker='o', color="red", label="Wishart interval")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("coverage.pdf")
    plt.savefig("coverage.png")


def run_coverage_test(confidence_level=0.95, method="normal", datasets=None, true_cov=None):
    """
    Run coverage test using pre-generated datasets.

    Args:
        confidence_level: Confidence level for intervals
        method: Method for calculating intervals ("normal", "wishart", etc.)
        datasets: List of pre-generated data arrays
        true_cov: True covariance matrix used to generate the data
    """
    n_features = true_cov.shape[0]
    n_toys = len(datasets)
    n_accept = np.zeros((n_features, n_features))

    for data in datasets:
        # calculate the covariance
        covariance1 = calc_sample_covariance(data)

        # Calculate covariance and confidence intervals
        covariance, covariance_lower, covariance_upper = calc_covariance_intervals(data=data, covariance=covariance1, confidence_level=confidence_level, method=method)

        # Check coverage
        accepts = np.where( (covariance_lower < true_cov) & (true_cov < covariance_upper), 1, 0)
        n_accept += accepts

    coverage = n_accept / n_toys
    return coverage


def test_invwishart_precision_interval():
    # Use default_rng instead of np.random.seed for cross-platform reproducibility.
    # The legacy RandomState API can produce different results across platforms/NumPy versions.
    # np.random.seed(42)
    rng = np.random.default_rng(42)

    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    true_precision = np.linalg.inv(true_cov)
    data = rng.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)

    # Calculate confidence interval
    # Use random_state for reproducible Monte Carlo sampling in invwishart method
    confidence_level = 0.95
    method = "invwishart"
    precision, precision_lower, precision_upper = calc_precision_intervals(data=data, confidence_level=confidence_level, method=method, random_state=123)

    print("DEBUG: true_precision =")
    print(true_precision)
    print("DEBUG: precision =")
    print(precision)
    print("DEBUG: precision_lower =")
    print(precision_lower)
    print("DEBUG: precision_upper =")
    print(precision_upper)

    if IS_MACOS:
        ref_true_precision =  np.array([[119.9195, -31.3883, -14.0845],
                                        [-31.3883,  58.5513,   7.0423],
                                        [-14.0845,   7.0423,  35.2113]])

        ref_precision =       np.array([[121.9875, -27.4081, -12.0025],
                                        [-27.4081,  55.1484,   5.6767],
                                        [-12.0025,   5.6767,  33.6392]])

        ref_precision_lower = np.array([[111.5952, -32.901,  -16.1478],
                                        [-32.901,   50.4932,   3.014 ],
                                        [-16.1478,   3.014,   30.8723]])

        ref_precision_upper = np.array([[133.2731, -22.2734,  -8.0867],
                                        [-22.2734,  60.1137,   8.4046],
                                        [ -8.0867,   8.4046,  36.7472]])

        assert np.allclose(true_precision, ref_true_precision, rtol=0, atol=1e-4)
        assert np.allclose(precision, ref_precision, rtol=0, atol=1e-3)
        assert np.allclose(precision_lower, ref_precision_lower, rtol=0, atol=1e-3)
        assert np.allclose(precision_upper, ref_precision_upper, rtol=0, atol=1e-3)

    else:
        ref_true_precision =  np.array([[119.9195, -31.3883, -14.0845],
                                        [-31.3883,  58.5513,   7.0423],
                                        [-14.0845,   7.0423,  35.2113]])

        ref_precision =       np.array([[121.1682, -26.7907, -15.5737],
                                        [-26.7907,  54.8474,   6.7126],
                                        [-15.5737,   6.7126,  34.7596]])

        ref_precision_lower = np.array([[110.8457, -32.2382, -19.8126],
                                        [-32.2382,  50.2181,   4.0009],
                                        [-19.8126,   4.0009,  31.8917]])

        ref_precision_upper = np.array([[132.378,  -21.7021, -11.5654],
                                        [-21.7021,  59.7716,   9.4871],
                                        [-11.5654,   9.4871,  37.9899]])

        assert np.allclose(true_precision, ref_true_precision, rtol=0, atol=1e-4)
        assert np.allclose(precision, ref_precision, rtol=0, atol=1e-3)
        assert np.allclose(precision_lower, ref_precision_lower, rtol=0, atol=1e-3)
        assert np.allclose(precision_upper, ref_precision_upper, rtol=0, atol=1e-3)


