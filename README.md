# covariance_calculators

[![CI badge](https://github.com/rreece/covariance_calculators/actions/workflows/ci.yml/badge.svg)](https://github.com/rreece/covariance_calculators/actions)

Implementations of covariance estimators and confidence intervals


## Introduction

This package provides numerically stable, online (streaming) estimators
for means, covariance matrices, and correlation matrices,
along with methods for computing confidence intervals on the covariance estimates.

Estimators included:

-   `OnlineCovariance` --- Welford's one-pass algorithm for incremental mean and covariance estimation, with support for merging independent estimators.
-   `EMACovariance` --- Exponential moving average covariance, parameterized by alpha, halflife, or span.
-   `SMACovariance` --- Simple moving average covariance over a fixed rolling window.

All estimators support a `geometric=True` mode that applies a log transform
to observations, suitable for multiplicative processes like financial returns,
and a `frequency` parameter for annualizing results (e.g. `frequency=252` for daily data).

Confidence interval methods for the sample covariance matrix include:

-   `asymptotic` --- Normal approximation using the Wishart variance formula.
-   `wishart` --- Monte Carlo sampling from the Wishart distribution.
-   `bootstrap` --- Bootstrap resampling.
-   `parametric_mc` --- Parametric Monte Carlo from a fitted normal.


## Basic usage

```python
import numpy as np
from covariance_calculators.estimators import OnlineCovariance
from covariance_calculators.intervals import calc_covariance_intervals

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

print("mean =", oc.mean)
print("cov =", oc.cov)
print("corr =", oc.corr)

## Compute 95% confidence intervals on the covariance estimate
cov, ci_lower, ci_upper = calc_covariance_intervals(
    data=data,
    confidence_level=0.95,
    method="asymptotic",
)
print("ci_lower =", ci_lower)
print("ci_upper =", ci_upper)
```

Example output:

```
mean = [ 0.0021 -0.0354 -0.0472]
cov = [[1.1017 0.6671 0.2329]
 [0.6671 2.0806 0.3757]
 [0.2329 0.3757 1.4245]]
corr = [[1.     0.4406 0.1859]
 [0.4406 1.     0.2182]
 [0.1859 0.2182 1.    ]]
ci_lower = [[0.965  0.5219 0.1211]
 [0.5219 1.8224 0.2211]
 [0.1211 0.2211 1.2478]]
ci_upper = [[1.2385 0.8122 0.3447]
 [0.8122 2.3388 0.5303]
 [0.3447 0.5303 1.6013]]
```


## Setup and running tests

Create and activate a virtual environment:

```bash
source setup.sh
```

This will create a `.venv` virtualenv (if one doesn't already exist),
install the dependencies from `requirements.txt`, and add the parent
directory to your `PYTHONPATH`.

Run the tests:

```bash
make test
```

You can also run the linter with:

```bash
make lint
```


## Coverage tests

Shows that the confidence intervals are well calibrated at 1, 2, 3, and 4 $\sigma$ confidence levels:

![Testing the statistical coverage of confidence intervals.](img/coverage.png)


## Theory of confidence intervals

### Cochran's theorem

For $n$ i.i.d. samples from a normal distribution, $x_i \sim N(\mu, \sigma^2)$,
Cochran's theorem gives the sampling distribution of the MLE variance estimator:

```math
\frac{n \hat{\sigma}^2}{\sigma^2} \sim \chi^{2}_{n-1}
```

where the MLEs for a normal distribution are

```math
\hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i
```

and

```math
\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} ( x_i - \hat{\mu} )^2
```

Note the unbiased sample variance is

```math
s^2 = \frac{1}{(n-1)} \sum_{i=1}^{n} ( x_i - \hat{\mu} )^2
```

So

```math
s^2 = \frac{n}{(n-1)} \hat{\sigma}^2
```

and

```math
\frac{(n-1) s^2}{\sigma^2} \sim \chi^{2}_{n-1}
```


### Quantiles to p-values

Cumulative distribution function:

```math
F(y) = \int_{-\infty}^{y} f(x) dx
```

```math
\bar{F}(y) = 1 - F(y) = \int_{y}^{\infty} f(x) dx
```

$p$-value from test statistic $q$:

```math
p = 1 - \alpha = \int_{-\infty}^{q_{\alpha}} f(q) dq = F(q(\alpha))
```

Critical value of test statistic for a given $p$-value:

```math
q_{\alpha} = F^{-1}(1 - \alpha) = \mathrm{ppf}(1 - \alpha)
```

Two sided:

```math
1 - \alpha = \int_{q_{\alpha}^\mathrm{lower}}^{q_{\alpha}^\mathrm{upper}} f(q) dq
```

```math
1 - \frac{\alpha}{2} = \int_{-\infty}^{q_{\alpha}^\mathrm{upper}} f(q) dq = F(q_{\alpha}^\mathrm{upper})
```

```math
1 - \frac{\alpha}{2} = \int_{q_{\alpha}^\mathrm{lower}}^{\infty} f(q) dq = \bar{F}(q_{\alpha}^\mathrm{lower}) = 1 - F(q_{\alpha}^\mathrm{lower})
```

```math
q_{\alpha}^\mathrm{upper} = F^{-1}(1 - \frac{\alpha}{2}) = \mathrm{ppf}(1 - \frac{\alpha}{2})
```

```math
q_{\alpha}^\mathrm{lower} = F^{-1}(\frac{\alpha}{2}) = \mathrm{ppf}(\frac{\alpha}{2})
```


### Confidence intervals for sample mean

Sample mean:

```math
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
```

Variance of sample mean:

```math
\mathrm{Var}(\bar{x}) = \mathrm{Var}\left( \frac{1}{n} \sum_{i=1}^{n} x_i \right) = \frac{1}{n^2} \sum_{i=1}^{n} \mathrm{Var}(x_i) = \frac{1}{n^2} \sum_{i=1}^{n} \sigma^2 = \frac{\sigma^2}{n} 
```

Asymptotically assuming the errors are normally distributed,
we have a frequentist confidence interval:

```math
\mu = \hat{\mu} \pm z_{\alpha} \sigma_{\hat{\mu}} = \bar{x} \pm z_\alpha \frac{\hat{\sigma}}{\sqrt{n}} \qquad \mathrm{at}~(1-\alpha)~\mathrm{CL}
```


### Wishart distribution

Scatter matrix:

```math
S = \sum_{i=1}^{n} ( x_i - \bar{x} ) ( x_i - \bar{x} )^\intercal
```

If the mean is known to be zero, $X \sim N_{p}(0, V)$, then $S \sim W_{p}(V, n)$.

If the mean is estimated from the data, $X \sim N_{p}(\mu, V)$, then $S \sim W_{p}(V, n-1)$.

If $p=1$ and $V=1$, then $W_{1}(1, n) = \chi^{2}_{n}$.

Variance of the $(i,j)$ element of $W \sim W_{p}(V, n)$:

```math
\mathrm{Var}(W_{ij}) = n \cdot ( V_{ii} V_{jj} + V_{ij}^{2} )
```


### Confidence intervals for sample covariance

Variance of the scatter matrix elements (from the Wishart with $n-1$ degrees of freedom):

```math
\mathrm{Var}(S_{ij}) = (n-1) ( V_{ii} V_{jj} + V_{ij}^{2} )
```

Sample covariance matrix:

```math
\hat{V} = \frac{1}{n-1} S
```

Variance of sample covariance matrix:

```math
\mathrm{Var}(\hat{V}) = \frac{1}{(n-1)^2} \mathrm{Var}(\hat{S}) = \frac{1}{n-1} ( \hat{V}_{ii} \hat{V}_{jj} + \hat{V}_{ij}^{2} )
```

Asymptotically assuming the errors are normally distributed:

```math
\hat{\sigma}_{ij} = \sqrt{\frac{1}{n-1} ( \hat{V}_{ii} \hat{V}_{jj} + \hat{V}_{ij}^{2} )}
```

and we have a frequentist confidence interval:

```math
V_{ij} = \hat{V}_{ij} \pm z_{\alpha} \hat{\sigma}_{ij}
```

at a confidence level picked by the two-sided $z$ score:

```math
z_{\alpha} = \Phi^{-1}\left(1 - \frac{\alpha}{2}\right)
```

because

```math
\Phi(z_{\alpha}) = \int_{-\infty}^{z_{\alpha}} \phi(x) dx  = 1 - \frac{\alpha}{2}
```

where

```math
\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}
```

Instead of using quantiles of the normal distribution we could use the quantiles of the Wishart distribution more directly.

```math
V_{ij} = \hat{V}_{ij} \pm \Delta_{ij}^{\alpha}
```

where $\Delta_{ij}^{\alpha}$ are determined by the quantiles of the Wishart distribution, $Q_{ij}$.

```math
1 - \frac{\alpha}{2} = F_{W}(Q^\mathrm{upper}; \hat{V}, n-1)
```

```math
Q^\mathrm{upper} = F_{W}^{-1}(1 - \frac{\alpha}{2}; \hat{V}, n-1)
```

```math
Q^\mathrm{lower} = F_{W}^{-1}(\frac{\alpha}{2}; \hat{V}, n-1)
```

```math
Q_{ij}^\mathrm{lower} < V_{ij} < Q_{ij}^\mathrm{upper} \qquad \mathrm{at}~(1-\alpha)~\mathrm{CL}
```

See also:

-   Quantiles of $\chi^2 \Rightarrow$ $p$-values [Table](https://math.arizona.edu/~jwatkins/chi-square-table.pdf)


## References

-   Cowan, G. (1998). *Statistical Data Analysis*.
-   James, F. (2006). *Statistical Methods in Experimental Particle Physics* (2nd ed.).
-   Heins, A. (2024). [Confidence intervals for Wishart random matrices](https://adamheins.com/blog/wishart-confidence-intervals).
-   [Cochran's theorem](https://en.wikipedia.org/wiki/Cochran%27s_theorem). *Wikipedia*.

