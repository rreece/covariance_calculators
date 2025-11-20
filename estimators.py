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
    def __init__(self, order, frequency=1, dtype=DEFAULT_DTYPE):
        """
        Parameters
        ----------
        order: int, The order (=="number of features") of the incrementally added
        dataset and of the resulting covariance matrix.
        """
        self._order = order
        self._count = 0
        self._mean = np.zeros(order, dtype=dtype)
        self._Cn = np.zeros((order, order), dtype=dtype)
        self.frequency = frequency
        self.dtype = dtype
        
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
    def cov(self):
        """
        array_like, The covariance matrix of the added data.
        Uses n-1 denominator for Bessel's correction.
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
        if isinstance(observation, np.ndarray):
            obs = observation
        elif isinstance(observation, list):
            obs = np.array(observation, dtype=self.dtype)
        elif isinstance(observation, int) or isinstance(observation, float):
            obs = np.array([observation], dtype=self.dtype)
        else:
            assert False

        if self._order != obs.size:
            raise ValueError(f"To add, observation size {obs.size} must be {self._order}")

        self._count += 1
        if self.count == 1: # First entry 
            self._mean = np.array(obs, dtype=self.dtype)
            return

        delta = obs - self._mean
        self._mean += delta / self.count
        delta_at_n = obs - self._mean

        self._Cn += np.outer(delta, delta_at_n)

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
    def __init__(self, order, alpha=None, halflife=None, span=None, frequency=1, dtype=DEFAULT_DTYPE):
        super(EMACovariance, self).__init__(order, frequency=frequency, dtype=dtype)
        _alpha = self._calc_alpha(alpha=alpha, halflife=halflife, span=span)
        assert 0 < _alpha < 1
        self._alpha = _alpha

    def _calc_alpha(self, alpha=None, halflife=None, span=None):
        if alpha:
            return alpha
        if halflife:
            return 1.0 - math.exp(-1*math.log(2)/halflife)
        if span:
            return 2.0 / (span + 1)

    @property
    def cov(self):
        """
        Note: denominator has W = 1
        """
        if self.count < 2:
            return None
        return self._Cn * self.frequency

    def add(self, observation):
        if isinstance(observation, np.ndarray):
            obs = observation
        elif isinstance(observation, list):
            obs = np.array(observation, dtype=self.dtype)
        elif isinstance(observation, int) or isinstance(observation, float):
            obs = np.array([observation], dtype=self.dtype)
        else:
            assert False

        if self._order != obs.size:
            raise ValueError(f"To add, observation size {obs.size} must be {self._order}")

        self._count += 1
        if self.count == 1: # First entry 
            self._mean = np.array(obs, dtype=self.dtype)
            return

        delta = obs - self._mean
        self._mean = (1 - self._alpha) * self._mean + self._alpha * obs
        delta_at_n = obs - self._mean

        if self.count == 2: # Second entry 
            self._Cn = np.outer(delta, delta_at_n)
            return

        self._Cn = (1 - self._alpha) * self._Cn + self._alpha * np.outer(delta, delta_at_n)

    def merge(self, other):
        """
        TODO: Update for EMACovariance
        """
        assert False, "Not implemented; Probably won't."


class SMACovariance(OnlineCovariance):
    """
    TODO
    """
    def __init__(self, order, span=None, frequency=1, dtype=DEFAULT_DTYPE):
        super(SMACovariance, self).__init__(order, frequency=frequency, dtype=dtype)
        self._span = span
        self.queue = list()

    def add(self, observation):
        """
        TODO: Test me
        """
        if isinstance(observation, np.ndarray):
            obs = observation
        elif isinstance(observation, list):
            obs = np.array(observation, dtype=self.dtype)
        elif isinstance(observation, int) or isinstance(observation, float):
            obs = np.array([observation], dtype=self.dtype)
        else:
            assert False

        if self._order != obs.size:
            raise ValueError(f"To add, observation size {obs.size} must be {self._order}")

        if self.count == self._span:
            assert len(self.queue) == self.count

            # pop oldest from queue
            obs_old = self.queue.pop(0)
            assert len(self.queue) == self._span - 1

        # add observation to queue
        self.queue.append(obs)
        self._count = len(self.queue)
        assert 1 <= self._count <= self._span

        # calc covariance over queue 
        # TODO: Would be better here to use a vectorized covariance calculation instead of online
        tmp_cov_calc = OnlineCovariance(self._order, frequency=self.frequency, dtype=self.dtype)
        for _obs in self.queue:
            tmp_cov_calc.add(_obs)

        # copy the state
        self._count = tmp_cov_calc._count
        self._mean = tmp_cov_calc._mean
        self._Cn = tmp_cov_calc._Cn


class RollingCovariance(OnlineCovariance):
    """
    Uses the Welford (1962) one-pass algorithm, also in reverse to
    remove observations that leave the rolling window.

    See:
    https://stackoverflow.com/questions/30876298/removing-a-prior-sample-while-using-welfords-method-for-computing-single-pass-v
    
    """
    def __init__(self, order, span=None, frequency=1, dtype=DEFAULT_DTYPE):
        super(RollingCovariance, self).__init__(order, frequency=frequency, dtype=dtype)
        self._span = span
        self.queue = list()

    def add(self, observation):
        """
        TODO: test me
        """
        if isinstance(observation, np.ndarray):
            obs = observation
        elif isinstance(observation, list):
            obs = np.array(observation, dtype=self.dtype)
        elif isinstance(observation, int) or isinstance(observation, float):
            obs = np.array([observation], dtype=self.dtype)
        else:
            assert False

        if self._order != obs.size:
            raise ValueError(f"To add, observation size {obs.size} must be {self._order}")

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

