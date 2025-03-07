"""
covariance_calculators/estimators.py

TODO:

[x] Sample mean and covariance
[x] Online mean and covariance
[ ] Exponential moving mean and covariance
[ ] Rolling mean and covariance
[ ] Shrinkage estimators
"""


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
#        self._shape = (order, order)
        self._identity = np.identity(order)
        self._ones = np.ones(order)
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
        return self._mean

    @property
    def cov(self):
        """
        array_like, The covariance matrix of the added data.
        """
        return self._Cn / (self._count + 1)

    @property
    def corr(self):
        """
        array_like, The normalized covariance matrix of the added data.
        Consists of the Pearson Correlation Coefficients of the data's features.
        """
        if self._count < 1:
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
        if self._order != len(observation):
            raise ValueError(f'Observation to add must be of size {self._order}')

        self._count += 1
        delta = observation - self._mean
        self._mean += delta / self._count
        delta_at_n = observation - self._mean
        self._Cn += np.outer(delta, delta_at_n)

#    def merge(self, other):
#        """
#        TODO: FIXME: _cov -> _Cn / _count
#
#        Merges the current object and the given other object into a new OnlineCovariance object.
#
#        Parameters
#        ----------
#        other: OnlineCovariance, The other OnlineCovariance to merge this object with.
#
#        Returns
#        -------
#        OnlineCovariance
#        """
#        if other._order != self._order:
#            raise ValueError(
#                   f'''
#                   Cannot merge two OnlineCovariances with different orders.
#                   ({self._order} != {other._order})
#                   ''')
#            
#        merged_cov = OnlineCovariance(self._order)
#        merged_cov._count = self.count + other.count
#        count_corr = (other.count * self.count) / merged_cov._count
#        merged_cov._mean = (self.mean/other.count + other.mean/self.count) * count_corr
#        flat_mean_diff = self._mean - other._mean
#        mean_diffs = np.broadcast_to(flat_mean_diff, (self._order, self._order)).T
#        merged_cov._cov = (self._cov * self.count \
#                           + other._cov * other._count \
#                           + mean_diffs * mean_diffs.T * count_corr) \
#                          / merged_cov.count
#        return merged_cov

