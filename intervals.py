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


def _prepare_data(data):
    """Return data as an array of shape (n_samples, n_features), or None."""
    if data is None:
        return None
    data = np.asarray(data)
    return data[:, np.newaxis] if data.ndim == 1 else data


def _calc_asymptotic_bounds(covariance, n_samples, alpha):
    """Return the normal-approximation (lower, upper) bounds for each element."""
    n_features = covariance.shape[0]

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
    return covariance - z_score * se_matrix, covariance + z_score * se_matrix


def calc_covariance_intervals(
    covariance = None,
    n_samples: int = None,
    data: np.ndarray = None,
    confidence_level: float = 0.95,
    method: str = 'asymptotic',
    random_state = None,
    n_draws: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample covariance matrix using unbiased estimators.

    Most methods are large-sample approximations that under-cover variances for
    small n_samples. 'chi2' is exact for variances of normal data; see README:
    "Coverage at small sample sizes".

    Parameters:
    -----------
    covariance : np.ndarray
        Input estimated covariance matrix (n_features, n_features)
    n_samples : int
        The number of samples used in estimation
    data : np.ndarray
        Input data matrix (n_samples, n_features), or (n_samples,) for one feature
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('asymptotic', 'chi2', 'wishart', 'bootstrap', or 'parametric_mc').
        'bootstrap' requires data.
    random_state : int, Generator, or RandomState, optional
        Seed or random state for reproducibility of Monte Carlo sampling
    n_draws : int, optional
        Monte Carlo draws for 'wishart' (default 10000), 'bootstrap' (default 1000),
        or 'parametric_mc' (default 2000). The quantiles are unreliable unless
        each tail gets several draws, so use many more than 2/(1 - confidence_level).

    Returns:
    --------
    covariance : np.ndarray
        Sample covariance matrix
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    """
    data = _prepare_data(data)

    if data is None and (covariance is None or n_samples is None):
        raise ValueError("Provide data, or both covariance and n_samples")

    if n_samples is None:
        n_samples = data.shape[0]

    if covariance is None:
        covariance = calc_sample_covariance(data)

    covariance = np.atleast_2d(covariance)
    n_features = covariance.shape[0]

    alpha = 1.0 - confidence_level

    if method == 'asymptotic':
        ci_lower, ci_upper = _calc_asymptotic_bounds(covariance, n_samples, alpha)

    elif method == 'chi2':
        # Off-diagonal elements have no simple exact interval, so use the asymptotic ones.
        ci_lower, ci_upper = _calc_asymptotic_bounds(covariance, n_samples, alpha)

        # Variances are exact for normal data, because dof * V_hat_ii / V_ii ~ chi2(dof).
        # See README: "Coverage at small sample sizes".
        dof = n_samples - 1
        scatter = dof * np.diagonal(covariance)
        diagonal = np.arange(n_features)
        ci_lower[diagonal, diagonal] = scatter / stats.chi2.ppf(1 - alpha/2, dof)
        ci_upper[diagonal, diagonal] = scatter / stats.chi2.ppf(alpha/2, dof)

    elif method == 'wishart':
        # Monte Carlo from the Wishart distribution (see README: "Wishart distribution").
        # S ~ W_p(V, n-1), and V_hat = S/(n-1). scipy's wishart(df=nu, scale=Sigma)
        # has E[W] = nu * Sigma. Setting scale = V_hat/(n-1) gives E[W] = V_hat,
        # so the samples are directly in covariance-matrix units.
        dof = n_samples - 1
        wishart_dist = stats.wishart(df=dof, scale=covariance/dof)

        n_draws = 10000 if n_draws is None else n_draws
        wishart_samples = wishart_dist.rvs(n_draws, random_state=random_state)
        wishart_samples = wishart_samples.reshape(n_draws, n_features, n_features)

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
        if data is None:
            raise ValueError("method='bootstrap' requires data")
        rng = np.random.default_rng(random_state)
        n_draws = 1000 if n_draws is None else n_draws
        bootstrap_covs = np.zeros((n_draws, n_features, n_features))

        for i in range(n_draws):
            bootstrap_indices = rng.choice(
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
        n_draws = 2000 if n_draws is None else n_draws
        mc_covs = np.zeros((n_draws, n_features, n_features))

        for i in range(n_draws):
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
        raise ValueError("Method must be 'asymptotic', 'chi2', 'wishart', 'bootstrap', or 'parametric_mc'")
        
    return covariance, ci_lower, ci_upper


def calc_precision_intervals(
    precision=None,
    n_samples: int = None,
    data: np.ndarray = None,
    confidence_level: float = 0.95,
    method: str = 'invwishart',
    random_state=None,
    n_draws: int = None,
    bias: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample precision matrix.

    The inverse of the unbiased covariance estimate is biased high by a factor of
    dof / (dof - p - 1), where dof = n_samples - 1 and p = n_features. By default
    this is corrected, so the returned precision is unbiased for normal data, and
    is not the exact inverse of the sample covariance. See README:
    "Bias of the precision estimate".

    Parameters:
    -----------
    precision : np.ndarray
        Inverse of the estimated covariance matrix (n_features, n_features)
    n_samples : int
        The number of samples used in estimation
    data : np.ndarray
        Input data matrix (n_samples, n_features), or (n_samples,) for one feature
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('invwishart')
    random_state : int, Generator, or RandomState, optional
        Seed or random state for reproducibility of Monte Carlo sampling
    n_draws : int, optional
        Monte Carlo draws (default 10000). The quantiles are unreliable unless
        each tail gets several draws, so use many more than 2/(1 - confidence_level).
    bias : bool
        If False (default), apply the bias correction to the precision and center
        the intervals on the result. If True, return the uncorrected inverse.

    Returns:
    --------
    precision : np.ndarray
        Precision matrix estimate
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    """
    data = _prepare_data(data)

    if data is None and (precision is None or n_samples is None):
        raise ValueError("Provide data, or both precision and n_samples")

    if n_samples is None:
        n_samples = data.shape[0]

    if precision is None:
        covariance = calc_sample_covariance(data)
        precision = np.linalg.inv(covariance)

    precision = np.atleast_2d(precision)
    n_features = precision.shape[0]

    alpha = 1.0 - confidence_level
    dof = n_samples - 1

    if method == 'invwishart':
        if dof - n_features - 1 <= 0:
            raise ValueError(f"method='invwishart' requires n_samples > n_features + 2, "
                             f"got n_samples={n_samples}, n_features={n_features}")

        if not bias:
            precision = precision * (dof - n_features - 1) / dof

        # Initialize inverse Wishart with correct scale, so its mean is precision
        invwishart_dist = stats.invwishart(df=dof, scale=precision * (dof-n_features-1))

        # Generate samples to estimate quantiles
        n_draws = 10000 if n_draws is None else n_draws
        invwishart_samples = invwishart_dist.rvs(n_draws, random_state=random_state)
        invwishart_samples = invwishart_samples.reshape(n_draws, n_features, n_features)

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

