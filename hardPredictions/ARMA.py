"""

ARMA Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains ARMA models using SciPy's minimization method.

Examples
-------------------------------------------------------------------------------


>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = ARMA(p = 2, q = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.predict(ts)
>>> prediction = model.forecast(ts, periods = 3)
>>> prediction
            ci_inf  ci_sup       series
1972-10-01     NaN     NaN  3646.138084
1972-11-01     NaN     NaN  4763.476525
1972-12-01     NaN     NaN  4205.279228
>>> prediction = model.forecast(ts, periods = 2, confidence_interval = 0.95)
>>> prediction
                ci_inf       ci_sup       series
1972-10-01   80.935482  6479.458676  3645.453875
1972-11-01  733.457614  8092.782317  4762.201208

"""

from base_model import base_model

import numpy
import scipy
import pandas
import matplotlib
from extras import add_next_date
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample

class ARMA(base_model):
    """ Moving-average model
    
    Parameter optimization method: scipy's minimization

    Args:
        p (int): AR order
        q (int): MA order

    Returns:
        ARMA model structure of order q.

    """

    def __init__(self, p=None, q=None, intercept=None, phi=None, theta=None):
        self.y = None
        
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p
        
        if q == None:
            raise ValueError('Please insert parameter q')
        else:
            self.q = q
        
        if intercept == None:
            self.intercept = None
        elif intercept == False:
            self.intercept = 0
        else:
            self.intercept = intercept
        
        if phi == None:
            self.phi = None
        else:
            self.phi = phi
            
        if theta == None:
            self.theta = None
        else:
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
        return 'ARMA(p = ' + str(self.p) + ', q = ' + str(self.q) + ', intercept = ' + str(self.intercept) + ', phi = ' + str(self.phi) + ', theta = ' + str(self.theta) +')'
        

    def params2vector(self):
        """ Parameters to vector
        
        Args:
            None.
            
        Returns:
            Vector parameters of length p+1 to use in optimization.

        """        
        params = list()
        if self.intercept == None:
            self.intercept = numpy.random.rand(1)[0]
        if self.phi == None:
            self.phi = numpy.random.rand(self.p)
        if self.theta == None:
            self.theta = numpy.random.rand(self.q)
        
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

    def __forward__(self, ts):
        if self.y == None:
            lon = len(ts.values)
            y = numpy.random.randn(lon)
        else:
            y = self.y
        lon = len(y)
        lon_ts = len(ts)
        if lon_ts <= self.p:
            ts_last = ts[0:lon]
            p_sum = numpy.dot(ts_last, self.phi[0:lon])
        else:
            ts_last = ts[lon-self.p:lon]
            p_sum = numpy.dot(ts_last, self.phi)
        if lon <= self.q:
            y_last = y[0:lon]                  
            q_sum = numpy.dot(y_last, self.theta[0:lon])
        else:
            y_last = y[lon-self.q:lon]
            q_sum = numpy.dot(y_last, self.theta)        

        result = self.intercept + p_sum + q_sum

        return result

    def predict(self, ts):
        """ Fits a time series using self model parameters
        
        Args:
            ts (pandas.Series): Time series to fit.
        
        Returns:
            Fitted time series.
            
        """

        prediction = list()
        for i in range(len(ts)):
            if i == 0:
                result = self.intercept
            else:
                result = self.__forward__(ts[0:i])
            prediction.append(result)
        prediction = pandas.Series((v for v in prediction), index = ts.index)
        return prediction


    def fit(self, ts, error_function = None):
        """ Finds optimal parameters using a given optimization function
        
        Args:
            ts (pandas.Series): Time series to fit.
            error_function (function): Function to estimates error.
            
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
    
    def simulate(self, ts, periods = 5, confidence_interval = 0.95, iterations = 1000):
        values = self.filter_ts(ts).values
        results = list()
        for i in range(iterations):

            for j in range(periods):
                train = resample(values, n_samples = 1)

                if j == 0:
                    y = ts
                else:
                    y = add_next_date(y, next_value_bootstrap)

                next_value = self.__forward__(y)
                next_value_bootstrap = next_value + train[0]
                result_complete = add_next_date(y, next_value_bootstrap)
                result = result_complete[-periods:]

            results.append(result)

        results = pandas.DataFrame(results)
        ci_inf = results.quantile(1-confidence_interval)
        ci_sup = results.quantile(confidence_interval)
        ci = pandas.DataFrame([ci_inf, ci_sup], index = ['ci_inf', 'ci_sup'])

        return ci

    def forecast(self, ts, periods, confidence_interval = None, iterations = 300):
        """ Predicts future values in a given period
        
        Args:
            ts (pandas.Series): Time series to predict.
            periods (int): Number of periods ahead to predict.
            
        Returns:
            Time series of predicted values.
        
        """
        for i in range(periods):
            if i == 0:
                y = ts

            value = self.__forward__(y)
            y = add_next_date(y, value)
        
        if confidence_interval == None:
            for i in range(periods):
                if i == 0:
                    ci_zero = ts
                ci_zero = add_next_date(ci_zero, None)
            
            ci_inf = ci_zero[-periods:]
            ci_sup = ci_zero[-periods:]
            ci = pandas.DataFrame([ci_inf, ci_sup], index = ['ci_inf', 'ci_sup'])            
        else:
            ci = self.simulate(ts, periods, confidence_interval, iterations)
            
        prediction = y[-periods:]
        prediction.name = 'series'
        result = ci.append(prediction)

        return result.transpose()
    
    def plot(self, ts, periods = 5, confidence_interval = None, iterations = 300):
        last = ts[-1:]
        fitted_ts = self.predict(ts)
        if periods == False:
            pass
        else:
            forecast_ts = self.forecast(ts, periods, confidence_interval, iterations)
            ci_inf = last.append(forecast_ts['ci_inf'])
            ci_sup = last.append(forecast_ts['ci_sup'])
            tseries = last.append(forecast_ts['series'])
        
        if periods == False:
            matplotlib.pyplot.plot(ts, 'k-')
            matplotlib.pyplot.plot(fitted_ts, 'b-')
            matplotlib.pyplot.legend(['Real', 'Fitted'])
        else:
            matplotlib.pyplot.plot(ts, 'k-')
            matplotlib.pyplot.plot(fitted_ts, 'c-')
            matplotlib.pyplot.plot(tseries, 'b-')
            matplotlib.pyplot.plot(ci_inf, 'r--')
            matplotlib.pyplot.plot(ci_sup, 'r--')
            matplotlib.pyplot.axvline(x = ts[-1:].index, color = 'k', linestyle = '--')
        
            if confidence_interval != None:
                matplotlib.pyplot.legend(['Real', 'Fitted', 'Forecast', 'CI', 'CI'])
            else:
                matplotlib.pyplot.legend(['Real', 'Fitted', 'Forecast'])

    def cross_validation(self, ts, n_splits, error_function = None):
        X = numpy.array(self.__get_X__(ts))
        y = numpy.array(ts.values.tolist())
        y_index = numpy.array(ts.index)
        tscv = TimeSeriesSplit(n_splits = n_splits)
        splits = tscv.split(X)

        error_list = list()
        for train_index, test_index in splits:
            y_train, y_test = y[train_index], y[test_index]
            y_train_index, y_test_index = y_index[train_index], y_index[test_index]

            y_train = pandas.Series((v for v in y_train), index = y_train_index)
            y_test = pandas.Series((v for v in y_test), index = y_test_index)
            #self.fit(y_train)
            error = self.calc_error(y_test, error_function)
            error_list.append(error)

        return error_list


    def get_predict_ci(self, ts, confidence_interval = 0.95, iterations = 1000):
        values = self.filter_ts(ts).values
        serie = self.predict(ts).values
        results = list()
        for i in range(iterations):
            result = list()
            for j in range(len(serie)):
                train = resample(values, n_samples = 1)
                new_value = train[0] + serie[j]
                result.append(new_value)

            results.append(result)

        results = pandas.DataFrame(results)
        minim = results.quantile(1-confidence_interval)
        maxim = results.quantile(confidence_interval)
        final_result = pandas.DataFrame([minim, maxim])

        return final_result