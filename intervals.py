"""
covariance_calculators/intervals.py

Initially based on generated versions in conversation with
Claude AI: Claude 3.5 Sonnet
Feb 1, 2025
"""

import numpy as np
from scipy import stats
from typing import Tuple

from covariance_calculators.estimators import calc_sample_covariance


def calc_covariance_intervals(
    covariance = None,
    n_samples: int = None,
    data: np.ndarray = None,
    confidence_level: float = 0.95,
    method: str = 'asymptotic',
    random_state = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample covariance matrix using unbiased estimators.
    
    Parameters:
    -----------
    covariance : np.ndarray
        Input estimated covariance matrix (n_features, n_features)
    n_samples : int
        The number of samples used in estimation
    data : np.ndarray
        Input data matrix (n_samples, n_features)
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('asymptotic', 'wishart', 'bootstrap', or 'parametric_mc')
    random_state : int, Generator, or RandomState, optional
        Seed or random state for reproducibility of Monte Carlo sampling
        
    Returns:
    --------
    covariance : np.ndarray
        Sample covariance matrix
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    """
    n_features = None

    if n_samples is None:
        n_samples, n_features = data.shape

    if covariance is None:
        covariance = calc_sample_covariance(data)

    if n_features is None:
        n_features, _ = covariance.shape

    assert not covariance is None
    assert not n_samples is None
    assert not n_features is None

    alpha = 1.0 - confidence_level

    if method == 'asymptotic':
        # Two-sided z score: z_alpha = Phi^{-1}(1 - alpha/2)
        # See README: "Confidence intervals for sample covariance"
        z_score = stats.norm.ppf(1 - alpha/2)
        se_matrix = np.zeros((n_features, n_features))

        # Standard error from the Wishart variance formula (see README):
        #   Var(V_ij) = (V_ii * V_jj + V_ij^2) / (n-1)
        #   se_ij = sqrt(Var(V_ij))
        for i in range(n_features):
            for j in range(n_features):
                se_matrix[i,j] = np.sqrt(
                    (covariance[i,i] * covariance[j,j] +
                     covariance[i,j]**2) / (n_samples - 1)
                )

        # Asymptotic CI: V_ij = V_hat_ij +/- z_alpha * se_ij
        ci_lower = covariance - z_score * se_matrix
        ci_upper = covariance + z_score * se_matrix

    elif method == 'wishart':
        # Monte Carlo from the Wishart distribution (see README: "Wishart distribution").
        # S ~ W_p(V, n-1), and V_hat = S/(n-1). scipy's wishart(df=nu, scale=Sigma)
        # has E[W] = nu * Sigma. Setting scale = V_hat/(n-1) gives E[W] = V_hat,
        # so the samples are directly in covariance-matrix units.
        dof = n_samples - 1
        wishart_dist = stats.wishart(df=dof, scale=covariance/dof)

        n_samples_wishart = 10000
        wishart_samples = wishart_dist.rvs(n_samples_wishart)

        # Element-wise quantiles: Q_ij^lower, Q_ij^upper (see README)
        ci_lower = np.zeros((n_features, n_features))
        ci_upper = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                samples_ij = wishart_samples[:, i, j]
                ci_lower[i, j] = np.percentile(samples_ij, 100 * alpha/2)
                ci_upper[i, j] = np.percentile(samples_ij, 100 * (1 - alpha/2))
        
    elif method == 'bootstrap':
        # Bootstrap approach (np.cov already uses n-1 by default)
        n_bootstrap = 1000
        bootstrap_covs = np.zeros((n_bootstrap, n_features, n_features))
        
        for i in range(n_bootstrap):
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            bootstrap_sample = data[bootstrap_indices]
            bootstrap_covs[i] = np.cov(bootstrap_sample, rowvar=False, bias=False)
        
        # Calculate percentile intervals
        ci_lower = np.percentile(bootstrap_covs, 100 * alpha/2, axis=0)
        ci_upper = np.percentile(bootstrap_covs, 100 * (1 - alpha/2), axis=0)

    elif method == 'parametric_mc':
        # Parametric Monte Carlo: generate fresh samples from N(0, estimated_cov)
        # This should converge to Wishart results for normal data
        rng = np.random.default_rng(random_state)
        n_mc = 2000
        mc_covs = np.zeros((n_mc, n_features, n_features))

        for i in range(n_mc):
            # Generate fresh sample from fitted multivariate normal
            mc_sample = rng.multivariate_normal(
                mean=np.zeros(n_features),
                cov=covariance,
                size=n_samples
            )
            mc_covs[i] = np.cov(mc_sample, rowvar=False, bias=False)

        ci_lower = np.percentile(mc_covs, 100 * alpha/2, axis=0)
        ci_upper = np.percentile(mc_covs, 100 * (1 - alpha/2), axis=0)

    else:
        raise ValueError("Method must be 'asymptotic', 'wishart', 'bootstrap', or 'parametric_mc'")
        
    return covariance, ci_lower, ci_upper


def calc_precision_intervals(
    precision=None,
    n_samples: int = None,
    data: np.ndarray = None,
    confidence_level: float = 0.95,
    method: str = 'invwishart',
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample precision matrix.

    Parameters:
    -----------
    precision : np.ndarray
        Input estimated precision matrix (n_features, n_features)
    n_samples : int
        The number of samples used in estimation
    data : np.ndarray
        Input data matrix (n_samples, n_features)
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('invwishart')
    random_state : int, Generator, or RandomState, optional
        Seed or random state for reproducibility of Monte Carlo sampling

    Returns:
    --------
    precision : np.ndarray
        Sample precision matrix
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    """
    n_features = None
    precision = None

    if n_samples is None:
        n_samples, n_features = data.shape

    if precision is None:
        covariance = calc_sample_covariance(data)
        precision = np.linalg.inv(covariance)

    if n_features is None:
        n_features, _ = precision.shape

    assert not precision is None
    assert not n_samples is None
    assert not n_features is None

    alpha = 1.0 - confidence_level
    dof = n_samples - 1

    if method == 'invwishart':
        # Initialize inverse Wishart with correct scale
        invwishart_dist = stats.invwishart(df=dof, scale=precision * (dof-n_features-1))

        # Generate samples to estimate quantiles
        n_samples_invwishart = 10000
        invwishart_samples = invwishart_dist.rvs(n_samples_invwishart, random_state=random_state)

        # Calculate element-wise quantiles (ppf)
        ci_lower = np.zeros((n_features, n_features))
        ci_upper = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                samples_ij = invwishart_samples[:, i, j]
                ci_lower[i, j] = np.percentile(samples_ij, 100 * alpha/2)
                ci_upper[i, j] = np.percentile(samples_ij, 100 * (1 - alpha/2))

    else:
        raise ValueError("Method must be 'invwishart'")

    return precision, ci_lower, ci_upper

