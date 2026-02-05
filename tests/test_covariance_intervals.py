"""
test_covariance_intervals.py
"""

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm

from covariance_calculators.estimators import calc_sample_covariance
from covariance_calculators.intervals import calc_covariance_intervals, calc_precision_intervals


mpl.use("Agg")
np.set_printoptions(precision=4, suppress=True)

IS_MACOS = sys.platform == "darwin"


def test_asymptotic_covariance_interval():
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
    method = "asymptotic"
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
    methods = ["asymptotic", "wishart", "bootstrap", "parametric_mc"]
    compare_methods(data, confidence_level=confidence_level, methods=methods)


def compare_methods(
    data: np.ndarray,
    confidence_level: float = 0.95,
    methods = None,
):
    """
    Compare asymptotic, Wishart, and bootstrap methods.
    """
    # Default to all methods
    if methods is None:
        methods = ["asymptotic", "wishart", "bootstrap", "parametric_mc"]

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

    assert np.allclose(results["asymptotic"][0], results["wishart"][0], rtol=0, atol=1e-4)
    assert np.allclose(results["asymptotic"][0], results["bootstrap"][0], rtol=0, atol=1e-4)

    return results


def generate_toy_datasets(n_toys, n_samples, true_cov, rng):
    """Pre-generate toy datasets for coverage testing."""
    n_features = true_cov.shape[0]
    datasets = []
    for _ in tqdm(range(n_toys), desc="Generating datasets"):
        data = rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=true_cov,
            size=n_samples
        )
        datasets.append(data)
    return datasets


def test_coverage(methods=None):
    """
    Test coverage of confidence intervals for different methods.

    Args:
        methods: List of methods to test. Options: 'asymptotic', 'wishart', 'bootstrap', 'parametric_mc'.
                 If None, tests all methods.
    """
    # Use default_rng instead of np.random.seed for cross-platform reproducibility.
    # The legacy RandomState API can produce different results across platforms/NumPy versions.
    # np.random.seed(42)
    rng = np.random.default_rng(42)
    # import hepplot to style plot
#    import hepplot as hep  # noqa

    # Default to all methods
    if methods is None:
#        methods = ["asymptotic", "wishart", "bootstrap", "parametric_mc"]
        methods = ["asymptotic", "wishart"]

    # Method configuration: n_toys, color, marker, label
    method_config = {
        "asymptotic": {
#            "n_toys": 500000,  # Many toys to probe 4-sigma (alpha ~ 6e-5, need ~500k for ~30 misses)
#            "n_toys": 100000,
            "n_toys": 10000,  # Fewer toys for faster test
            "color": "#1f77b4",
            "marker": "o",
            "label": "Asymptotic",
        },
        "wishart": {
#            "n_toys": 10000,  # 10k toys gives ~27 misses at 3-sigma, ~0.6 at 4-sigma
#            "n_toys": 1000,
            "n_toys": 100,  # Fewer toys for faster test
            "color": "red",
            "marker": "o",
            "label": "Wishart",
        },
        "bootstrap": {
#            "n_toys": 10000,
#            "n_toys": 1000,
            "n_toys": 100,  # Fewer toys for faster test
            "color": "green",
            "marker": "o",
            "label": "Bootstrap",
        },
        "parametric_mc": {
#            "n_toys": 10000,
#            "n_toys": 1000,
            "n_toys": 100,  # Fewer toys for faster test
            "color": "blueviolet",
            "marker": "o",
            "label": "Parametric MC",
        },
    }

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

    # Pre-generate datasets (use max n_toys needed across selected methods)
    max_n_toys = max(method_config[m]["n_toys"] for m in methods)
    datasets = generate_toy_datasets(max_n_toys, n_samples, true_cov, rng)

    # Run coverage tests for each method (all confidence levels in one pass)
    results = {}
    for method in methods:
        config = method_config[method]
        n_toys = config["n_toys"]
        method_datasets = datasets[:n_toys]

        print(f"\n=== {config['label']} ({n_toys} toys) ===")
        # Get coverage for all confidence levels in one pass through the data
        coverage_dict = run_coverage_test(
            confidence_levels=confidence_levels,
            method=method,
            datasets=method_datasets,
            true_cov=true_cov
        )

        # Convert to list of alphas in same order as confidence_levels
        coverage_alphas = [1.0 - np.average(coverage_dict[cl]) for cl in confidence_levels]
        print(coverage_alphas)
        results[method] = coverage_alphas

    # make coverage plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = 1 - p_\mathrm{CL}$")
    ax.set_ylabel(r"$\alpha_\mathrm{coverage} = 1 - p_\mathrm{coverage}$")
    ax.plot(alphas, alphas, color="darkgray", label="Perfect calibration")

    for method in methods:
        config = method_config[method]
        ax.plot(alphas, results[method], marker=config["marker"], color=config["color"], label=config["label"])
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("coverage.pdf")
    plt.savefig("coverage.png")


def run_coverage_test(confidence_levels, method="asymptotic", datasets=None, true_cov=None):
    """
    Run coverage test using pre-generated datasets for multiple confidence levels in one pass.

    Args:
        confidence_levels: List of confidence levels for intervals
        method: Method for calculating intervals ("asymptotic", "wishart", etc.)
        datasets: List of pre-generated data arrays
        true_cov: True covariance matrix used to generate the data

    Returns:
        Dict mapping confidence_level -> coverage matrix
    """
    n_features = true_cov.shape[0]
    n_toys = len(datasets)

    # Initialize acceptance counts for each confidence level
    n_accept = {cl: np.zeros((n_features, n_features)) for cl in confidence_levels}

    for data in tqdm(datasets, desc=f"{method}", leave=False):
        # calculate the covariance once per dataset
        covariance1 = calc_sample_covariance(data)

        # Calculate intervals and check coverage for all confidence levels
        for cl in confidence_levels:
            covariance, covariance_lower, covariance_upper = calc_covariance_intervals(
                data=data, covariance=covariance1, confidence_level=cl, method=method
            )
            accepts = np.where((covariance_lower < true_cov) & (true_cov < covariance_upper), 1, 0)
            n_accept[cl] += accepts

    # Convert counts to coverage fractions
    coverage = {cl: n_accept[cl] / n_toys for cl in confidence_levels}
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


