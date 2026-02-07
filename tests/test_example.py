"""
test_example.py
"""

import numpy as np

from covariance_calculators.estimators import OnlineCovariance
from covariance_calculators.intervals import calc_covariance_intervals


np.set_printoptions(precision=4, suppress=True)


def test_example():
    ## Generate toy data: 500 samples from a 3d normal distribution
    rng = np.random.default_rng(42)
    true_cov = np.array([[1.0, 0.5, 0.2],
                         [0.5, 2.0, 0.3],
                         [0.2, 0.3, 1.5]])
    L = np.linalg.cholesky(true_cov)
    data = rng.standard_normal((500, 3)) @ L.T

    ## Stream data through the online estimator
    oc = OnlineCovariance(order=3)
    for row in data:
        oc.add(row)

    print("mean =")
    print(oc.mean)
    print("cov =")
    print(oc.cov)
    print("corr =")
    print(oc.corr)

    # Check estimates are close to truth
    assert np.allclose(oc.mean, 0, atol=0.1)
    assert np.allclose(oc.cov, true_cov, atol=0.3)

    ## Compute 95% confidence intervals on the covariance estimate
    cov, ci_lower, ci_upper = calc_covariance_intervals(
        data=data,
        confidence_level=0.95,
        method="asymptotic",
    )
    print("ci_lower =")
    print(ci_lower)
    print("ci_upper =")
    print(ci_upper)

    # Check estimates are close to the true covariance
    assert np.allclose(cov, true_cov, atol=0.3)

    # Check that the true covariance is mostly within the confidence intervals
    covered = (ci_lower <= true_cov) & (true_cov <= ci_upper)
    assert np.sum(covered) >= 7

    # Check against known deterministic output (seed=42, n=500)
    expected_ci_lower = np.array([[0.965,  0.5219, 0.1211],
                                  [0.5219, 1.8224, 0.2211],
                                  [0.1211, 0.2211, 1.2478]])
    expected_ci_upper = np.array([[1.2385, 0.8122, 0.3447],
                                  [0.8122, 2.3388, 0.5303],
                                  [0.3447, 0.5303, 1.6013]])
    assert np.allclose(ci_lower, expected_ci_lower, atol=1e-4)
    assert np.allclose(ci_upper, expected_ci_upper, atol=1e-4)


if __name__ == "__main__":
    test_example()
