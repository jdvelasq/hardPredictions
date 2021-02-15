"""

SARIMA Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains SARIMA model based on SARIMA statmodel's one.

Examples
-------------------------------------------------------------------------------

Get predicted values as a DataFrame:

Load time series
>>> ts = load_champagne()

>>> model = SARIMA(p = 3, d = 1, q = 3, P = 3, D = 1, Q = 3, s = 1)
>>> model
SARIMA(p = 3, d = 1, q = 3, P = 3, D = 1, Q = 3, s = 1)

>>> random.seed(1)
>>> model.fit(ts)
SARIMA(p = 3, d = 1, q = 3, P = 3, D = 1, Q = 3, s = 1)

>>> random.seed(1)
>>> model.predict(ts, periods = 2) # doctest: +ELLIPSIS
                 ci_inf        ci_sup  ...      forecast  real
1972-10-01  4445...  10945...  ...   7620...  None
1972-11-01  3914...  11549...  ...   7322...  None
<BLANKLINE>
[2 rows x 6 columns]


Classes
-------------------------------------------------------------------------------

"""

from skfore.base_model import base_model

import statsmodels



class SARIMA(base_model):
    """ Seasonal autoregressive integrated moving average model

    Parameter optimization method: scipy's minimization

    Args:
        p (int): AR order
        d (int): Integration order of the process
        q (int): MA order
        P (int): AR seasonal order
        D (int): Integration seasonal order of the process
        Q (int): MA seasonal order
        s (int): Periodicity


    Returns:
        SARIMA model structure of order (p,d,q)x(P,D,Q)s

    """

    def __init__(self, p=None, d=None, q=None, P=None, D=None, Q=None, s=None, **kwargs):

        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q

        if s == None:
            self.s = 0
        else:
            self.s = s

        self.model = None



    def __repr__(self):
        return 'SARIMA(p = ' + str(self.p) + ', d = ' + str(self.d) + ', q = ' + str(self.q) + ', P = ' + str(self.P) + ', D = ' + str(self.D) + ', Q = ' + str(self.Q) +', s = ' + str(self.s) +')'


    def forecast(self, y):
        """ Next step

        Args:
            ts (pandas.Series): Time series to find next value

        Returns:
            Value of next time stamp

        """

        result = self.model.predict(start=len(y), end=len(y))

        return result.values[0]

    def simulate(self, ts):
        """ Fits a time series using self model parameters

        Args:
            ts (pandas.Series): Time series to fit

        Returns:
            Fitted time series

        """
        y = ts
        prediction = list()
        for i in range(len(y)):
            result = self.forecast(y[0:i])
            prediction.append(result)
        prediction = pandas.Series((v for v in prediction), index = ts.index)
        return prediction



    def fit(self, ts, error_function = None, **kwargs):
        """ Finds optimal parameters using a given optimization function

        Args:
            ts (pandas.Series): Time series to fit
            error_function (function): Function to estimates error

        Return:
            self

        """

        if self.p == None:
            raise ValueError('Please insert parameter p')

        if self.d == None:
            raise ValueError('Please insert parameter d')

        if self.q == None:
            raise ValueError('Please insert parameter q')

        if self.P == None:
            raise ValueError('Please insert parameter P')

        if self.D == None:
            raise ValueError('Please insert parameter D')

        if self.Q == None:
            raise ValueError('Please insert parameter Q')

        try:
            self.model = statsmodels.tsa.statespace.sarimax.SARIMAX(ts, order = (self.p,self.d,self.q), seasonal_order = (self.P,self.D,self.Q,self.s), **kwargs)
            self.model = self.model.fit()
        except:
            self.model = statsmodels.tsa.statespace.sarimax.SARIMAX(ts, order = (self.p,self.d,self.q), seasonal_order = (self.P,self.D,self.Q,self.s), enforce_invertibility = False, **kwargs)
            self.model = self.model.fit()

        return self

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.ELLIPSIS)
