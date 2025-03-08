"""
covariance_calculators/estimators.py

TODO:

[x] Sample mean and covariance
[x] Online mean and covariance
[x] Exponential moving mean and covariance
[ ] Rolling mean and covariance
[ ] Shrinkage estimators
"""


import math
import numpy as np


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

    Adapted from:
    https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html
    """
    def __init__(self, order):
        """
        Parameters
        ----------
        order: int, The order (=="number of features") of the incrementally added
        dataset and of the resulting covariance matrix.
        """
        self._order = order
        self._count = 0
        self._mean = np.zeros(order)
        self._Cn = np.zeros((order, order))
        
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
        return self._mean

    @property
    def cov(self):
        """
        array_like, The covariance matrix of the added data.
        """
        if self.count < 2:
            return None
        return self._Cn / (self.count + 1)

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
            _obs = observation
        elif isinstance(observation, list):
            _obs = np.array(observation, dtype=float)
        elif isinstance(observation, int) or isinstance(observation, float):
            _obs = np.array([observation], dtype=float)
        else:
            assert False

        if self._order != _obs.size:
            raise ValueError(f"To add, observation size {_obs.size} must be {self._order}")

        self._count += 1
        if self.count == 1: # First entry 
            self._mean = np.array(_obs, dtype=float)
            return

        print("DEBUG: _obs = ", _obs)
        print("DEBUG: _mean = ", self._mean)
        delta = _obs - self._mean
        print("DEBUG: delta = ", delta)
        self._mean += delta / self.count
        delta_at_n = _obs - self._mean

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
    def __init__(self, order, alpha=None, halflife=None, span=None):
        super(EMACovariance, self).__init__(order)
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
        return self._Cn

    def add(self, observation):
        if isinstance(observation, np.ndarray):
            _obs = observation
        elif isinstance(observation, list):
            _obs = np.array(observation, dtype=float)
        elif isinstance(observation, int) or isinstance(observation, float):
            _obs = np.array([observation], dtype=float)
        else:
            assert False

        if self._order != _obs.size:
            raise ValueError(f"To add, observation size {_obs.size} must be {self._order}")

        self._count += 1
        if self.count == 1: # First entry 
            self._mean = np.array(_obs, dtype=float)
            return

        delta = _obs - self._mean
        self._mean = (1 - self._alpha) * self._mean + self._alpha * _obs
        delta_at_n = _obs - self._mean

        if self.count == 2: # Second entry 
            self._Cn = np.outer(delta, delta_at_n)
            return

        self._Cn = (1 - self._alpha) * self._Cn + self._alpha * np.outer(delta, delta_at_n)

    def merge(self, other):
        """
        TODO: Update for EMACovariance
        """
        assert False, "Not implemented; Probably won't."

