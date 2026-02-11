"""
covariance_calculators/estimators.py

TODO:

[x] Sample mean and covariance
[x] Online mean and covariance
[x] Exponential moving mean and covariance
[ ] NaN protection
[ ] Rolling mean and covariance
[ ] Shrinkage estimators
"""


import math
import numpy as np


DEFAULT_DTYPE = np.float64


def calc_sample_mean(data, frequency=None):
    _mean = np.mean(data, axis=0)
    if not frequency is None:
        _mean = _mean * frequency
    return _mean


def calc_sample_covariance(data, frequency=None):
    # V_hat = S/(n-1), the unbiased sample covariance (Bessel's correction).
    covariance = np.cov(data, rowvar=False, bias=False)  # bias=False uses n-1
    if not frequency is None:
        covariance = covariance * frequency
    return covariance


class OnlineCovariance:
    """
    A class to calculate the mean and the covariance matrix
    of the incrementally added, n-dimensional data.

    Uses the Welford (1962) one-pass algorithm.

    Adapted from:
    https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html
    """
    def __init__(self, order, frequency=1, geometric=False, dtype=DEFAULT_DTYPE):
        """
        Parameters
        ----------
        order: int, The order (=="number of features") of the incrementally added
        dataset and of the resulting covariance matrix.
        geometric: bool, If True, transform observations to log returns via log(1 + r)
        before processing. Gives CAGR-style mean and multiplicative-process covariance.
        """
        self._order = order
        self._count = 0
        self._mean = np.zeros(order, dtype=dtype)
        self._Cn = np.zeros((order, order), dtype=dtype)
        self.frequency = frequency
        self.geometric = geometric
        self.dtype = dtype
        
    def _prepare_observation(self, observation):
        """
        Convert observation to numpy array, validate size,
        and optionally apply log transform for geometric mode.
        """
        if isinstance(observation, np.ndarray):
            obs = observation
        elif isinstance(observation, list):
            obs = np.array(observation, dtype=self.dtype)
        elif isinstance(observation, (int, float)):
            obs = np.array([observation], dtype=self.dtype)
        else:
            assert False
        if self._order != obs.size:
            raise ValueError(f"To add, observation size {obs.size} must be {self._order}")
        if self.geometric:
            obs = np.log1p(obs)
        return obs

    @property
    def count(self):
        """
        int, The number of observations that has been added
        to this instance of OnlineCovariance.
        """
        return self._count
   
    @property
    def mean(self):
        """
        double, The mean of the added data.
        """
        if self.count < 1:
            return None
        return self._mean * self.frequency

    @property
    def cagr(self):
        """
        Compound Annual Growth Rate: exp(mean) - 1.
        Converts the annualized continuously compounded rate to CAGR.
        Only meaningful when geometric=True.
        """
        if self.mean is None:
            return None
        return np.expm1(self.mean)

    @property
    def mean_return(self):
        """
        Annualized mean return. Returns CAGR when geometric=True,
        arithmetic mean otherwise.
        """
        if self.geometric:
            return self.cagr
        return self.mean

    @property
    def cov(self):
        """
        array_like, The covariance matrix of the added data.
        V_hat = S/(n-1), using Bessel's correction (see README: "Wishart distribution").
        """
        if self.count < 2:
            return None
        return self._Cn * (self.frequency / (self.count - 1))

    @property
    def corr(self):
        """
        array_like, The normalized covariance matrix of the added data.
        Consists of the Pearson Correlation Coefficients of the data's features.
        """
        if self.count < 2:
            return None
        variances = np.diagonal(self.cov)
        denomiator = np.sqrt(variances[np.newaxis,:] * variances[:,np.newaxis])
        return self.cov / denomiator

    def add(self, observation):
        """
        Add the given observation to this object.

        Parameters
        ----------
        observation: array_like, The observation to add.
        """
        obs = self._prepare_observation(observation)

        self._count += 1
        if self.count == 1: # First entry
            self._mean = np.array(obs, dtype=self.dtype)
            return

        # Welford (1962) one-pass update for the scatter matrix
        # S = sum_i (x_i - x_bar)(x_i - x_bar)^T (see README: "Wishart distribution").
        # _Cn accumulates S incrementally: C_n = C_{n-1} + delta * delta'^T
        # where delta = x_n - x_bar_{n-1} and delta' = x_n - x_bar_n.
        delta = obs - self._mean
        self._mean += delta / self.count
        delta_at_n = obs - self._mean

        self._Cn += np.outer(delta, delta_at_n)

    def to_dict(self):
        """Serialize internal state to a plain dict (with numpy arrays as values)."""
        return {
            "class": "OnlineCovariance",
            "order": self._order,
            "count": self._count,
            "mean": self._mean.copy(),
            "Cn": self._Cn.copy(),
            "frequency": self.frequency,
            "geometric": self.geometric,
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, d):
        """Create an OnlineCovariance instance from a state dict."""
        dtype = np.dtype(d["dtype"])
        obj = cls(order=d["order"], frequency=d["frequency"],
                  geometric=d["geometric"], dtype=dtype)
        obj._count = int(d["count"])
        obj._mean = np.array(d["mean"], dtype=dtype)
        obj._Cn = np.array(d["Cn"], dtype=dtype)
        return obj

    def merge(self, other):
        """
        Merges the current object and the given other object into a new OnlineCovariance object.

        Parameters
        ----------
        other: OnlineCovariance, The other OnlineCovariance to merge this object with.

        Returns
        -------
        OnlineCovariance
        """
        if other._order != self._order:
            raise ValueError(
                   f"""
                   Cannot merge two OnlineCovariances with different orders.
                   ({self._order} != {other._order})
                   """)
            
        merged = OnlineCovariance(self._order)
        merged._count = self.count + other.count
        w_self = self.count/merged.count
        w_other = other.count/merged.count
        merged._mean = (self.mean * w_self) + (other.mean * w_other)
        delta_mean = self._mean - other._mean
        w_delta = (other.count * self.count) / merged.count
        merged._Cn = self._Cn + other._Cn + np.outer(delta_mean, delta_mean) * w_delta
        return merged


class EMACovariance(OnlineCovariance):
    def __init__(self, order, alpha=None, halflife=None, span=None, warmup=None, frequency=1, geometric=False, dtype=DEFAULT_DTYPE):
        super(EMACovariance, self).__init__(order, frequency=frequency, geometric=geometric, dtype=dtype)
        _alpha = self._calc_alpha(alpha=alpha, halflife=halflife, span=span)
        assert 0 < _alpha < 1
        self._alpha = _alpha
        # Warmup: accumulate first N observations and initialize with SMA
        # Default warmup is span // 5 (if provided), otherwise no warmup
        # Warmup must be 0 (disabled) or >= 2 (need at least 2 observations for covariance)
        if warmup is not None:
            assert warmup == 0 or warmup >= 2, f"warmup must be 0 or >= 2, got {warmup}"
            self._warmup = warmup
        elif span is not None:
            self._warmup = max(span // 5, 2)
        else:
            self._warmup = 0  # no warmup
        self._warmup_buffer = []

    def _calc_alpha(self, alpha=None, halflife=None, span=None):
        if alpha:
            return alpha
        if halflife:
            return 1.0 - math.exp(-1*math.log(2)/halflife)
        if span:
            return 2.0 / (span + 1)

    @property
    def mean(self):
        """
        Override to return None during warmup period.
        """
        if self.count < 1:
            return None
        # During warmup, return None (not enough data for stable estimate)
        if self._warmup > 0 and self._count <= self._warmup:
            return None
        return self._mean * self.frequency

    @property
    def cov(self):
        """
        Note: denominator has W = 1
        """
        if self.count < 2:
            return None
        # During warmup, return None (not enough data for stable estimate)
        if self._warmup > 0 and self._count <= self._warmup:
            return None
        return self._Cn * self.frequency

    def add(self, observation):
        obs = self._prepare_observation(observation)

        self._count += 1

        # Warmup phase: accumulate observations and initialize with SMA
        if self._warmup > 0 and self._count <= self._warmup:
            self._warmup_buffer.append(obs)
            if self._count == self._warmup:
                # Initialize mean and covariance from SMA of warmup period
                data = np.array(self._warmup_buffer, dtype=self.dtype)
                self._mean = np.mean(data, axis=0)
                if len(data) >= 2:
                    # Initialize _Cn with sample covariance scaled for EMA
                    # Use (n-1) * cov as the base, then scale by (1-alpha) to
                    # give appropriate weight relative to future EMA updates
                    sample_cov = np.cov(data, rowvar=False, ddof=1)
                    self._Cn = sample_cov * (1 - self._alpha)
                self._warmup_buffer = []  # free memory
            return

        # Normal EMA update
        if self._warmup == 0 and self.count == 1:
            # No warmup: first entry initializes mean (original behavior)
            self._mean = np.array(obs, dtype=self.dtype)
            return

        delta = obs - self._mean
        self._mean = (1 - self._alpha) * self._mean + self._alpha * obs
        delta_at_n = obs - self._mean

        if self._warmup == 0 and self.count == 2:
            # No warmup: second entry initializes covariance (original behavior)
            self._Cn = np.outer(delta, delta_at_n)
            return

        self._Cn = (1 - self._alpha) * self._Cn + self._alpha * np.outer(delta, delta_at_n)

    def to_dict(self):
        """Serialize internal state to a plain dict."""
        d = super().to_dict()
        d["class"] = "EMACovariance"
        d["alpha"] = self._alpha
        d["warmup"] = self._warmup
        d["warmup_buffer"] = [obs.copy() for obs in self._warmup_buffer]
        return d

    @classmethod
    def from_dict(cls, d):
        """Create an EMACovariance instance from a state dict."""
        dtype = np.dtype(d["dtype"])
        obj = cls(order=d["order"], alpha=d["alpha"], warmup=d["warmup"],
                  frequency=d["frequency"], geometric=d["geometric"], dtype=dtype)
        obj._count = int(d["count"])
        obj._mean = np.array(d["mean"], dtype=dtype)
        obj._Cn = np.array(d["Cn"], dtype=dtype)
        obj._warmup_buffer = [np.array(obs, dtype=dtype) for obs in d["warmup_buffer"]]
        return obj

    def merge(self, other):
        """
        TODO: Update for EMACovariance
        """
        assert False, "Not implemented; Probably won't."


class SMACovariance(OnlineCovariance):
    """
    TODO
    """
    def __init__(self, order, span=None, frequency=1, geometric=False, dtype=DEFAULT_DTYPE):
        super(SMACovariance, self).__init__(order, frequency=frequency, geometric=geometric, dtype=dtype)
        self._span = span
        self.queue = list()

    def add(self, observation):
        obs = self._prepare_observation(observation)

        if self.count == self._span:
            assert len(self.queue) == self.count

            # pop oldest from queue
            self.queue.pop(0)  # discard oldest
            assert len(self.queue) == self._span - 1

        # add observation to queue
        self.queue.append(obs)
        self._count = len(self.queue)
        assert 1 <= self._count <= self._span

        # calc covariance over queue using vectorized numpy operations
        # Convert queue to numpy array for efficient computation
        data = np.array(self.queue, dtype=self.dtype)

        # Compute mean (annualized by frequency)
        self._mean = np.mean(data, axis=0)

        # Compute scatter matrix S = sum (x_i - x_bar)(x_i - x_bar)^T.
        # _Cn stores S; the cov property divides by (n-1) to give V_hat.
        if self._count >= 2:
            self._Cn = np.cov(data, rowvar=False, ddof=1) * (self._count - 1)
        else:
            # Single observation: no covariance yet
            self._Cn = np.zeros((self._order, self._order), dtype=self.dtype)


    def to_dict(self):
        """Serialize internal state to a plain dict."""
        d = super().to_dict()
        d["class"] = "SMACovariance"
        d["span"] = self._span
        d["queue"] = [obs.copy() for obs in self.queue]
        return d

    @classmethod
    def from_dict(cls, d):
        """Create an SMACovariance instance from a state dict."""
        dtype = np.dtype(d["dtype"])
        obj = cls(order=d["order"], span=d["span"],
                  frequency=d["frequency"], geometric=d["geometric"], dtype=dtype)
        obj._count = int(d["count"])
        obj._mean = np.array(d["mean"], dtype=dtype)
        obj._Cn = np.array(d["Cn"], dtype=dtype)
        obj.queue = [np.array(obs, dtype=dtype) for obs in d["queue"]]
        return obj


class RollingCovariance(OnlineCovariance):
    """
    Uses the Welford (1962) one-pass algorithm, also in reverse to
    remove observations that leave the rolling window.

    See:
    https://stackoverflow.com/questions/30876298/removing-a-prior-sample-while-using-welfords-method-for-computing-single-pass-v

    TODO: Don't think this algorithm is numerically stable.
    Maybe we should just remove it.
    """
    def __init__(self, order, span=None, frequency=1, geometric=False, dtype=DEFAULT_DTYPE):
        super(RollingCovariance, self).__init__(order, frequency=frequency, geometric=geometric, dtype=dtype)
        self._span = span
        self.queue = list()

    def add(self, observation):
        obs = self._prepare_observation(observation)

        if self.count == self._span:
            assert len(self.queue) == self.count

            # pop oldest
            obs_old = self.queue.pop(0)
            assert len(self.queue) == self._span - 1

            delta_old_at_n = obs_old - self.mean
            self._mean -= (obs_old - self.mean)/(self.count - 1)
            delta_old = obs_old - self.mean
            self._Cn -= np.outer(delta_old, delta_old_at_n)

        self.queue.append(obs)
        self._count = len(self.queue)
        assert 1 <= self._count <= self._span

        if self.count == 1: # First entry 
            self._mean = np.array(obs, dtype=self.dtype)
            return

        delta = obs - self._mean
        self._mean += delta / self.count
        delta_at_n = obs - self._mean

        self._Cn += np.outer(delta, delta_at_n)

