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
        # Calculate standard errors using asymptotic formula with n-1 correction
        z_score = stats.norm.ppf(1 - alpha/2)
        se_matrix = np.zeros((n_features, n_features))
        
        # Standard error calculated from Wishart variance
        # Var(V_ij) = (V_ii V_jj + V_ij^2) / (n-1)
        # se_ij = sqrt( Var(V_ij) )
        for i in range(n_features):
            for j in range(n_features):
                se_matrix[i,j] = np.sqrt(
                    (covariance[i,i] * covariance[j,j] + 
                     covariance[i,j]**2) / (n_samples - 1)
                )
        
        ci_lower = covariance - z_score * se_matrix
        ci_upper = covariance + z_score * se_matrix

    elif method == 'wishart':
        # Using Wishart distribution
        # Initialize Wishart distribution with scale matrix S/(df)
        # Scale matrix is S/(df) because wishart.rvs returns W/df where W ~ W_p(df, scale)
        dof = n_samples - 1
        wishart_dist = stats.wishart(df=dof, scale=covariance/dof)

        # Generate samples to estimate quantiles
        n_samples_wishart = 10000
        wishart_samples = wishart_dist.rvs(n_samples_wishart)

        # Calculate element-wise quantiles (ppf)
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


def calc_tangent_portfolio(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """
    Calculate tangent portfolio weights (maximum Sharpe ratio portfolio).

    w ∝ Σ⁻¹(μ - r_f), normalized to sum to 1

    Parameters:
    -----------
    mean_returns : np.ndarray
        Expected returns for each asset (n_assets,)
    covariance : np.ndarray
        Covariance matrix of returns (n_assets, n_assets)
    risk_free_rate : float
        Risk-free rate (default: 0.0)

    Returns:
    --------
    weights : np.ndarray
        Tangent portfolio weights (sum to 1)
    """
    excess_returns = mean_returns - risk_free_rate
    precision = np.linalg.inv(covariance)
    raw_weights = precision @ excess_returns
    weights = raw_weights / np.sum(raw_weights)
    return weights


def calc_tangent_portfolio_intervals(
    data: np.ndarray,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state=None,
) -> dict:
    """
    Calculate confidence intervals for tangent portfolio weights and Sharpe ratio.

    Uses bootstrap resampling to propagate uncertainty from returns data
    through to portfolio weights and performance metrics.

    Parameters:
    -----------
    data : np.ndarray
        Returns data matrix (n_samples, n_assets). Each row is a time period,
        each column is an asset.
    risk_free_rate : float
        Risk-free rate per period (default: 0.0, assumes excess returns)
    confidence_level : float
        Confidence level (default: 0.95)
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)
    random_state : int, Generator, or RandomState, optional
        Seed or random state for reproducibility

    Returns:
    --------
    dict with keys:
        'weights': Point estimate of tangent portfolio weights
        'weights_ci_lower': Lower CI for weights
        'weights_ci_upper': Upper CI for weights
        'sharpe_ratio': Point estimate of Sharpe ratio
        'sharpe_ratio_ci_lower': Lower CI for Sharpe ratio
        'sharpe_ratio_ci_upper': Upper CI for Sharpe ratio
        'expected_return': Point estimate of portfolio expected return
        'volatility': Point estimate of portfolio volatility
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_assets = data.shape
    alpha = 1.0 - confidence_level

    # Point estimates
    mean_returns = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False, bias=False)
    weights = calc_tangent_portfolio(mean_returns, covariance, risk_free_rate)

    # Portfolio metrics
    portfolio_return = weights @ mean_returns
    portfolio_volatility = np.sqrt(weights @ covariance @ weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Bootstrap
    bootstrap_weights = np.zeros((n_bootstrap, n_assets))
    bootstrap_sharpe = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_data = data[indices]

        # Compute bootstrap estimates
        boot_mean = np.mean(boot_data, axis=0)
        boot_cov = np.cov(boot_data, rowvar=False, bias=False)

        try:
            boot_weights = calc_tangent_portfolio(boot_mean, boot_cov, risk_free_rate)
            boot_port_return = boot_weights @ boot_mean
            boot_port_vol = np.sqrt(boot_weights @ boot_cov @ boot_weights)
            boot_sharpe = (boot_port_return - risk_free_rate) / boot_port_vol

            bootstrap_weights[i] = boot_weights
            bootstrap_sharpe[i] = boot_sharpe
        except np.linalg.LinAlgError:
            # Singular matrix - use NaN
            bootstrap_weights[i] = np.nan
            bootstrap_sharpe[i] = np.nan

    # Compute confidence intervals (ignoring NaN)
    weights_ci_lower = np.nanpercentile(bootstrap_weights, 100 * alpha / 2, axis=0)
    weights_ci_upper = np.nanpercentile(bootstrap_weights, 100 * (1 - alpha / 2), axis=0)
    sharpe_ci_lower = np.nanpercentile(bootstrap_sharpe, 100 * alpha / 2)
    sharpe_ci_upper = np.nanpercentile(bootstrap_sharpe, 100 * (1 - alpha / 2))

    return {
        'weights': weights,
        'weights_ci_lower': weights_ci_lower,
        'weights_ci_upper': weights_ci_upper,
        'sharpe_ratio': sharpe_ratio,
        'sharpe_ratio_ci_lower': sharpe_ci_lower,
        'sharpe_ratio_ci_upper': sharpe_ci_upper,
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
    }

