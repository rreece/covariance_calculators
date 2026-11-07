"""
test_covariance_intervals.py
"""

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tqdm import tqdm

from covariance_calculators.estimators import calc_sample_covariance
from covariance_calculators.intervals import calc_covariance_intervals, calc_precision_intervals


mpl.use("Agg")
np.set_printoptions(precision=4, suppress=True)



def generate_data(rng, true_cov, n_samples):
    """
    Generate zero-mean normal data using a Cholesky factor. Unlike
    rng.multivariate_normal, whose SVD depends on the platform's LAPACK,
    this gives the same data on every platform for a given seed.
    """
    L = np.linalg.cholesky(true_cov)
    return rng.standard_normal((n_samples, true_cov.shape[0])) @ L.T


def test_asymptotic_covariance_interval():
    # Use default_rng instead of np.random.seed for cross-platform reproducibility.
    # The legacy RandomState API can produce different results across platforms/NumPy versions.
    # np.random.seed(42)
    rng = np.random.default_rng(42)

    # Generate sample data
    n_samples = 1000
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = generate_data(rng, true_cov, n_samples)

    # Calculate confidence interval
    confidence_level = 0.95
    method = "asymptotic"
    covariance, covariance_lower, covariance_upper = calc_covariance_intervals(data=data, confidence_level=confidence_level, method=method)

    ref_covariance = np.array([[ 0.0103,  0.0052,  0.0026],
                               [ 0.0052,  0.0201, -0.0008],
                               [ 0.0026, -0.0008,  0.0298]])

    assert np.allclose(covariance, ref_covariance, rtol=0, atol=1e-4)
    assert np.allclose(covariance, true_cov, rtol=0, atol=3e-3)


def test_compare_methods():
    # Use default_rng instead of np.random.seed for cross-platform reproducibility.
    # The legacy RandomState API can produce different results across platforms/NumPy versions.
    # np.random.seed(42)
    rng = np.random.default_rng(42)

    # Generate sample data
    n_samples = 1000
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = generate_data(rng, true_cov, n_samples)

    # Compare all methods
    confidence_level = 0.95
    methods = ["asymptotic", "chi2", "wishart", "bootstrap", "parametric_mc"]
    compare_methods(data, confidence_level=confidence_level, methods=methods)


def compare_methods(
    data: np.ndarray,
    confidence_level: float = 0.95,
    methods = None,
):
    """
    Compare asymptotic, chi2, Wishart, and bootstrap methods.
    """
    # Default to all methods
    if methods is None:
        methods = ["asymptotic", "chi2", "wishart", "bootstrap", "parametric_mc"]

    # calculate the covariance once for all methods
    covariance1 = calc_sample_covariance(data)

    results = dict()
    for method in methods:
        results[method] = calc_covariance_intervals(data=data, covariance=covariance1, confidence_level=confidence_level, method=method, random_state=7)

    # Print results
    for method in methods:
        print(f"\n{method} method estimate:")
        covariance, covariance_lower, covariance_upper = results[method]
        print(covariance)
        print("lower:")
        print(covariance_lower)
        print("upper:")
        print(covariance_upper)

    # At large n, every method's bounds should agree to within 10% of the asymptotic width
    _, ref_lower, ref_upper = results["asymptotic"]
    tolerance = 0.1 * (ref_upper - ref_lower)
    for method in methods:
        _, lower, upper = results[method]
        assert np.all(np.abs(lower - ref_lower) < tolerance), method
        assert np.all(np.abs(upper - ref_upper) < tolerance), method

    return results


def generate_toy_datasets(n_toys, n_samples, true_cov, rng):
    """Pre-generate toy datasets for coverage testing."""
    return [generate_data(rng, true_cov, n_samples)
            for _ in tqdm(range(n_toys), desc="Generating datasets")]


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
    plt.close(fig)

    # Measured alpha should be close to nominal. The tolerances allow for the
    # toy statistics of each method. 4 sigma is not checked because 10k toys
    # are too few to resolve it.
    relative_tolerances = {
        "asymptotic": [0.05, 0.10, 0.30],
        "wishart": [0.20, 0.70],
    }
    for method in methods:
        for i, tolerance in enumerate(relative_tolerances.get(method, [])):
            assert abs(results[method][i] / alphas[i] - 1) < tolerance, (method, i)


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

    for i_toy, data in enumerate(tqdm(datasets, desc=f"{method}", leave=False)):
        # calculate the covariance once per dataset
        covariance1 = calc_sample_covariance(data)

        # Calculate intervals and check coverage for all confidence levels
        for cl in confidence_levels:
            covariance, covariance_lower, covariance_upper = calc_covariance_intervals(
                data=data, covariance=covariance1, confidence_level=cl, method=method, random_state=i_toy
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
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    true_precision = np.linalg.inv(true_cov)
    data = generate_data(rng, true_cov, n_samples)

    # Calculate confidence interval
    # Use random_state for reproducible Monte Carlo sampling in invwishart method
    confidence_level = 0.95
    method = "invwishart"
    precision, precision_lower, precision_upper = calc_precision_intervals(data=data, confidence_level=confidence_level, method=method, random_state=123, bias=True)

    print("DEBUG: true_precision =")
    print(true_precision)
    print("DEBUG: precision =")
    print(precision)
    print("DEBUG: precision_lower =")
    print(precision_lower)
    print("DEBUG: precision_upper =")
    print(precision_upper)

    ref_true_precision =  np.array([[119.9195, -31.3883, -14.0845],
                                    [-31.3883,  58.5513,   7.0423],
                                    [-14.0845,   7.0423,  35.2113]])

    ref_precision =       np.array([[115.1627, -30.1826, -10.7038],
                                    [-30.1826,  57.8084,   4.1089],
                                    [-10.7038,   4.1089,  34.5969]])

    ref_precision_lower = np.array([[105.3518, -35.7258, -14.7556],
                                    [-35.7258,  52.9322,   1.3503],
                                    [-14.7556,   1.3503,  31.7463]])

    ref_precision_upper = np.array([[125.8169, -25.0206,  -6.8327],
                                    [-25.0206,  63.032,    6.9034],
                                    [ -6.8327,   6.9034,  37.8032]])

    assert np.allclose(true_precision, ref_true_precision, rtol=0, atol=1e-4)
    assert np.allclose(precision, ref_precision, rtol=0, atol=1e-3)
    assert np.allclose(precision_lower, ref_precision_lower, rtol=0, atol=1e-3)
    assert np.allclose(precision_upper, ref_precision_upper, rtol=0, atol=1e-3)


MC_METHODS = ["wishart", "bootstrap", "parametric_mc"]
ALL_METHODS = ["asymptotic", "chi2"] + MC_METHODS


@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 2)) * np.array([1.0, 2.0])


@pytest.mark.parametrize("method", MC_METHODS)
def test_random_state_reproducible(small_data, method):
    """The same seed should repeat the intervals, and a different seed should not."""
    kwargs = {"data": small_data, "method": method, "n_draws": 200}

    first = calc_covariance_intervals(random_state=1, **kwargs)
    repeat = calc_covariance_intervals(random_state=1, **kwargs)
    other = calc_covariance_intervals(random_state=2, **kwargs)

    assert np.array_equal(first[1], repeat[1])
    assert np.array_equal(first[2], repeat[2])
    assert not np.array_equal(first[1], other[1])


def test_precision_random_state_reproducible(small_data):
    first = calc_precision_intervals(data=small_data, random_state=1, n_draws=200)
    repeat = calc_precision_intervals(data=small_data, random_state=1, n_draws=200)

    assert np.array_equal(first[1], repeat[1])


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("shape", [(100, 1), (100,)])
def test_single_feature_intervals(method, shape):
    """One feature works as a column or a 1-d array, with results of shape (1, 1)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(shape)

    covariance, lower, upper = calc_covariance_intervals(data=data, method=method, random_state=1)

    assert covariance.shape == lower.shape == upper.shape == (1, 1)
    assert np.isclose(covariance[0, 0], np.var(data, ddof=1))
    assert lower[0, 0] < covariance[0, 0] < upper[0, 0]


def test_single_feature_covariance_input():
    covariance, lower, upper = calc_covariance_intervals(
        covariance=np.array([[2.0]]), n_samples=50, method="wishart", random_state=1)

    assert covariance.shape == lower.shape == upper.shape == (1, 1)
    assert lower[0, 0] < 2.0 < upper[0, 0]


def test_single_feature_precision_intervals():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 1))

    precision, lower, upper = calc_precision_intervals(data=data, random_state=1)

    assert precision.shape == lower.shape == upper.shape == (1, 1)
    assert lower[0, 0] < precision[0, 0] < upper[0, 0]


def test_precision_argument_is_used(small_data):
    """A given precision matrix should be used, with or without data."""
    precision = np.array([[7.0, 1.0],
                          [1.0, 5.0]])

    without_data, lower, upper = calc_precision_intervals(precision=precision, n_samples=100, random_state=1, bias=True)
    with_data, _, _ = calc_precision_intervals(precision=precision, data=small_data, random_state=1, bias=True)

    assert np.array_equal(without_data, precision)
    assert np.array_equal(with_data, precision)
    assert np.all((lower < precision) & (precision < upper))


def test_precision_requires_enough_samples():
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="n_samples"):
        calc_precision_intervals(data=rng.standard_normal((5, 3)))


def test_missing_inputs_raise():
    with pytest.raises(ValueError, match="Provide data"):
        calc_covariance_intervals()
    with pytest.raises(ValueError, match="Provide data"):
        calc_covariance_intervals(covariance=np.eye(2))
    with pytest.raises(ValueError, match="Provide data"):
        calc_precision_intervals(precision=np.eye(2))
    with pytest.raises(ValueError, match="requires data"):
        calc_covariance_intervals(covariance=np.eye(2), n_samples=50, method="bootstrap")


def test_n_draws_reaches_tail_quantiles(small_data):
    """Enough draws should push a 4-sigma bound beyond the default draws' sample extreme."""
    alpha = 6.3e-5
    kwargs = {"data": small_data, "confidence_level": 1 - alpha, "method": "wishart", "random_state": 1}

    _, _, upper_few = calc_covariance_intervals(n_draws=100, **kwargs)
    _, _, upper_many = calc_covariance_intervals(n_draws=200000, **kwargs)

    assert upper_many[0, 0] > upper_few[0, 0]


# --- chi2 method tests ---


def test_chi2_variance_bounds_match_formula():
    """For n=10 and V=1, the bounds use the published chi2 quantiles for 9 dof: 19.023 and 2.700."""
    _, lower, upper = calc_covariance_intervals(covariance=np.eye(2), n_samples=10, method="chi2")

    assert np.allclose(np.diag(lower), 9 / 19.023, rtol=0, atol=1e-3)
    assert np.allclose(np.diag(upper), 9 / 2.700, rtol=0, atol=1e-3)


def test_chi2_off_diagonal_matches_asymptotic():
    covariance = np.array([[1.0, 0.5],
                           [0.5, 2.0]])

    _, chi2_lower, chi2_upper = calc_covariance_intervals(covariance=covariance, n_samples=50, method="chi2")
    _, asym_lower, asym_upper = calc_covariance_intervals(covariance=covariance, n_samples=50, method="asymptotic")

    assert chi2_lower[0, 1] == chi2_lower[1, 0] == asym_lower[0, 1]
    assert chi2_upper[0, 1] == chi2_upper[1, 0] == asym_upper[0, 1]


def test_chi2_does_not_modify_input():
    covariance = np.array([[1.0, 0.5],
                           [0.5, 2.0]])
    original = covariance.copy()

    returned, _, _ = calc_covariance_intervals(covariance=covariance, n_samples=50, method="chi2")

    assert np.array_equal(covariance, original)
    assert np.array_equal(returned, original)


def test_chi2_covers_variances_at_small_n():
    """At n=10, chi2 should reach 95% coverage with equal tails, unlike asymptotic."""
    rng = np.random.default_rng(42)

    n_toys = 4000
    n_samples = 10
    true_cov = np.array([[1.0, 0.5],
                         [0.5, 2.0]])
    diagonal = np.eye(2, dtype=bool)

    n_above = {"chi2": np.zeros((2, 2)), "asymptotic": np.zeros((2, 2))}
    n_below = {"chi2": np.zeros((2, 2)), "asymptotic": np.zeros((2, 2))}

    for _ in range(n_toys):
        data = generate_data(rng, true_cov, n_samples)
        covariance = calc_sample_covariance(data)
        for method in n_above:
            _, lower, upper = calc_covariance_intervals(covariance=covariance, n_samples=n_samples, method=method)
            n_above[method] += true_cov >= upper
            n_below[method] += true_cov <= lower

    assert np.allclose(n_above["chi2"][diagonal] / n_toys, 0.025, rtol=0, atol=0.01)
    assert np.allclose(n_below["chi2"][diagonal] / n_toys, 0.025, rtol=0, atol=0.01)
    assert np.all(n_above["asymptotic"][diagonal] / n_toys > 0.08)


# --- precision bias tests ---


def test_precision_bias_correction_factor(small_data):
    """The default should rescale the uncorrected estimate and bounds by (dof - p - 1) / dof."""
    biased = calc_precision_intervals(data=small_data, random_state=1, n_draws=500, bias=True)
    unbiased = calc_precision_intervals(data=small_data, random_state=1, n_draws=500)

    dof = 99
    factor = (dof - 2 - 1) / dof

    for result, reference in zip(unbiased, biased):
        assert np.allclose(result, reference * factor, rtol=1e-9)


def test_precision_correction_does_not_modify_input():
    precision = np.array([[7.0, 1.0],
                          [1.0, 5.0]])
    original = precision.copy()

    calc_precision_intervals(precision=precision, n_samples=100, random_state=1, n_draws=100)

    assert np.array_equal(precision, original)


def test_precision_unbiased_with_good_coverage_by_default():
    """With few samples per feature, only the corrected default should be unbiased and covered."""
    rng = np.random.default_rng(42)

    n_toys = 200
    n_samples = 20
    n_features = 10
    A = rng.standard_normal((n_features, 2 * n_features))
    true_cov = A @ A.T / (2 * n_features) + 0.5 * np.eye(n_features)
    true_precision = np.linalg.inv(true_cov)
    diagonal = np.eye(n_features, dtype=bool)

    totals = {False: np.zeros(n_features), True: np.zeros(n_features)}
    covered = {False: np.zeros(n_features), True: np.zeros(n_features)}

    for i_toy in range(n_toys):
        data = generate_data(rng, true_cov, n_samples)
        for bias in totals:
            precision, lower, upper = calc_precision_intervals(data=data, random_state=i_toy, n_draws=1000, bias=bias)
            totals[bias] += precision[diagonal]
            covered[bias] += (lower[diagonal] < true_precision[diagonal]) & (true_precision[diagonal] < upper[diagonal])

    assert np.allclose(totals[False] / n_toys / true_precision[diagonal], 1.0, rtol=0, atol=0.15)
    assert np.isclose(np.mean(covered[False]) / n_toys, 0.95, rtol=0, atol=0.03)
    assert np.all(totals[True] / n_toys / true_precision[diagonal] > 1.5)
    assert np.mean(covered[True]) / n_toys < 0.8
