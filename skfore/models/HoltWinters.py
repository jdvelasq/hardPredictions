"""

Holt Winters Model
===============================================================================

Overview
-------------------------------------------------------------------------------



Classes
-------------------------------------------------------------------------------

"""

from skfore.models.BaseModel import BaseModel

import statsmodels



class HoltWinters(BaseModel):
    """ Autoregressive model

    Parameter optimization method: scipy's minimization

    Args:
        p (int): order
        intercept (boolean or double): False for set intercept to 0 or double
        phi (array): array of p-length for set parameters without optimization

    Returns:
        AR model structure of order p

    """

    def __init__(self, **kwargs):


        self.model = None



    def __repr__(self):
        return 'HoltWinters(trend = ' + str(4) +')'


    def forecast(self, y):

        result = self.model.simulate(start=len(y), end=len(y))

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



        self.model = statsmodels.tsa.holtwinters.ExponentialSmoothing(ts, **kwargs)
        self.model = self.model.fit()

        return self
