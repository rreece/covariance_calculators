"""
covariance_calculators/intervals.py

Initially based on generated versions in conversation with
Claude AI: Claude 3.5 Sonnet
Feb 1, 2025
"""

import math
import numpy as np
from scipy import stats
from typing import Tuple, Optional

from covariance_calculators.estimators import calc_sample_covariance


def calc_covariance_intervals(
    data: np.ndarray,
    covariance=None,
    confidence_level: float = 0.95,
    method: str = 'normal',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample covariance matrix using unbiased estimators.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix (n_samples, n_features)
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('normal', 'wishart', or 'bootstrap')
        
    Returns:
    --------
    covariance : np.ndarray
        Sample covariance matrix
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    """
    n_samples, n_features = data.shape

    if covariance is None:
        covariance = calc_sample_covariance(data)

    alpha = 1.0 - confidence_level

    if method == 'normal':
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
    
    else:
        raise ValueError("Method must be either 'normal', 'wishart', or 'bootstrap'")
        
    return covariance, ci_lower, ci_upper


def calc_precision_intervals(
    data: np.ndarray,
    precision=None,
    confidence_level: float = 0.95,
    method: str = 'invwishart',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample precision matrix.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix (n_samples, n_features)
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('invwishart')
        
    Returns:
    --------
    precision : np.ndarray
        Sample precision matrix
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    """
    n_samples, n_features = data.shape

    if precision is None:
        covariance = calc_sample_covariance(data)
        precision = np.linalg.inv(covariance)

    alpha = 1.0 - confidence_level
    dof = n_samples - 1

    if method == 'invwishart':
        # Initialize inverse Wishart with correct scale
        invwishart_dist = stats.invwishart(df=dof, scale=precision * (dof-n_features-1))

        # Generate samples to estimate quantiles
        n_samples_invwishart = 10000
        invwishart_samples = invwishart_dist.rvs(n_samples_invwishart)

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

