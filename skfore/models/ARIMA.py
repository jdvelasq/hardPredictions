"""

ARIMA Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains ARIMA model based on skfore ARMA models.

Examples
-------------------------------------------------------------------------------

Get predicted values as a DataFrame:

Load time series
>>> ts = load_champagne()

>>> model = ARIMA(p = 3, d = 1, q = 3)
>>> model
ARIMA(p = 3, d = 1, q = 3, intercept = None, phi = None, theta = None)

>>> random.seed(1)
>>> model.fit(ts) # doctest: +ELLIPSIS
ARIMA(p = 3, d = 1, q = 3, intercept = 158..., phi = [-0.401...  0.227...  0.294...], theta = [ 0.008... -0.932... -0.275...])

>>> random.seed(1)
>>> model.predict(ts, periods = 2) # doctest: +ELLIPSIS
                 ci_inf        ci_sup  ...      forecast  real
1972-10-01  3390...   9592...  ...   6380...  None
1972-11-01  3541...  10152...  ...   6696...  None
<BLANKLINE>
[2 rows x 6 columns]


"""

from skfore.base_model import base_model

from sklearn import *


class ARIMA(base_model):
    """ Autoregressive integrated moving average model

    Parameter optimization method: scipy's minimization

    Args:
        p (int): AR order
        d (int): I
        q (int): MA order
        intercept (boolean or double): False for set intercept to 0 or double
        phi (array): array of p-length for set parameters without optimization
        theta (array): array of q-length for set parameters without optimization

    Returns:
        ARIMA model structure of order p,d,q

    """

    def __init__(self, p=None, d=None, q=None, intercept=None, phi=None, theta=None):
        self.y = None

        if p == None:
            self.p = 0
        else:
            self.p = p

        if d == None:
            self.d = 0
        else:
            self.d = d

        if q == None:
            self.q = 0
        else:
            self.q = q

        if intercept == False:
            self.intercept = 0
        else:
            self.intercept = intercept

        self.phi = phi
        self.theta = theta

        if intercept == None and theta == None and phi == None:
            self.optim_type = 'complete'
        elif intercept == None and theta != None and phi != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and theta == None and phi == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and theta == None and phi == None:
            self.optim_type = 'optim_params'
        elif intercept != None and theta != None and phi != None:
            self.optim_type = 'no_optim'
        else:
            raise ValueError('Please fulfill all parameters')


    def __repr__(self):
        return 'ARIMA(p = ' + str(self.p) + ', d = ' + str(self.d) + ', q = ' + str(self.q) + ', intercept = ' + str(self.intercept) + ', phi = ' + str(self.phi) + ', theta = ' + str(self.theta) +')'


    def params2vector(self):
        """ Parameters to vector

        Args:
            None

        Returns:
            Vector parameters of length p+q+1 to use in optimization

        """
        params = list()

        if self.intercept == None:
            self.intercept = numpy.random.rand(1)[0]
        if self.phi == None:
            self.phi = numpy.random.rand(self.p)
        if self.theta== None:
            self.theta = numpy.random.rand(self.p)

        if self.optim_type == 'complete':
            params.append(self.intercept)
            for i in range(len(self.phi)):
                params.append(self.phi[i])
            for i in range(len(self.theta)):
                params.append(self.theta[i])
            return params
        elif self.optim_type == 'no_intercept' or self.optim_type == 'optim_params':
            for i in range(len(self.phi)):
                params.append(self.phi[i])
            for i in range(len(self.theta)):
                params.append(self.theta[i])
            return params
        elif self.optim_type == 'optim_intercept':
            params.append(self.intercept)
            return params
        elif self.optim_type == 'no_optim':
            pass


    def vector2params(self, vector):
        """ Vector to parameters

        Args:
            vector (list): vector of length p+1 to convert into parameters of
            the model

        Returns:
            self

        """

        if self.optim_type == 'complete':
            self.intercept = vector[0]
            self.phi = vector[1:self.p + 1]
            self.theta = vector[self.p + 1:]
        elif self.optim_type == 'no_intercept' or self.optim_type == 'optim_params':
            self.phi = vector[0:self.p]
            self.theta = vector[self.p:]
        elif self.optim_type == 'optim_intercept':
            self.intercept = vector[0]
        elif self.optim_type == 'no_optim':
            pass

        return self

    def forecast(self, ts):
        """ Next step

        Args:
            ts (pandas.Series): Time series to find next value

        Returns:
            Value of next time stamp

        """
        lon_ts = len(ts.values)

        if self.d == 0:
            diff = 0
        elif lon_ts <= self.d:
            diff = 0
        else:
            diff = ts.values[-1] - ts.values[-self.d]

        if self.p == 0:
            p_sum = 0
        elif lon_ts <= self.p:
            ts_last = ts.values[0:lon_ts]
            p_sum = self.intercept + numpy.dot(ts_last, self.phi[0:lon_ts])
        else:
            ts_last = ts.values[lon_ts-self.p:lon_ts]
            p_sum = self.intercept + numpy.dot(ts_last, self.phi)

        if self.q == 0:
            q_sum = 0
        else:
            history = list()
            predictions = list()
            for t in numpy.arange(0,lon_ts,1):
                length = len(history)

                if length <= self.q:
                    yhat = numpy.mean(ts.values[0:t])
                else:
                    ts_last = history[length-self.q:length]
                    predicted = predictions[length-self.q:length]
                    mean_predicted = numpy.mean(ts_last)
                    new_predicted = self.intercept + numpy.dot(numpy.subtract(ts_last, predicted), self.theta)
                    yhat = mean_predicted + new_predicted

                predictions.append(yhat)
                history.append(ts.values[t])

            if lon_ts == 1:
                q_sum = ts.values[0]
            elif lon_ts <= self.q:
                q_sum = numpy.mean(history[0:lon_ts])
            else:
                ts_last = history[lon_ts-self.q:lon_ts]
                predicted = predictions[lon_ts-self.q:lon_ts]
                mean_predicted = numpy.mean(ts_last)
                new_predicted = self.intercept + numpy.dot(numpy.subtract(ts_last, predicted), self.theta)
                q_sum = mean_predicted + new_predicted

        result = -diff + p_sum + q_sum

        return result

    def simulate(self, ts):
        """ Fits a time series using self model parameters

        Args:
            ts (pandas.Series): Time series to fit

        Returns:
            Fitted time series

        """

        prediction = list()
        for i in range(len(ts)):
            if i == 0:
                try:
                    result = ts[0]
                except:
                    result = 0
            else:
                result = self.forecast(ts[0:i])
            prediction.append(result)
        prediction = pandas.Series((v for v in prediction), index = ts.index)
        return prediction


    def fit(self, ts, error_function = None):
        """ Finds optimal parameters using a given optimization function

        Args:
            ts (pandas.Series): Time series to fit
            error_function (function): Function to estimates error

        Return:
            self

        """

        if self.optim_type == 'no_optim':
            pass
        else:
            def f(x):
                self.vector2params(x)
                return self.calc_error(ts, error_function)

            x0 = self.params2vector()
            optim_params = scipy.optimize.minimize(f, x0)
            self.vector2params(vector = optim_params.x)

        return self

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.ELLIPSIS)
