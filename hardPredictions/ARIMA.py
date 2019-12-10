"""

ARIMA Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains ARIMA model based on hardPredictions ARMA models.

Examples
-------------------------------------------------------------------------------


>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)

"""

from base_model import base_model

import numpy
import scipy
import pandas
from hardPredictions.extras import add_next_date

class ARIMA(base_model):
    """ Moving-average model
    
    Parameter optimization method: scipy's minimization

    Args:
        p (int): AR order
        d (int): I
        q (int): MA order

    Returns:

    """

    def __init__(self, p=None, d=None, q=None, intercept=None, phi=None, theta=None):
        self.y = None
        
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p
            
        if d == None:
            raise ValueError('Please insert parameter d')
        else:
            self.d = d
        
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
        return 'ARIMA(p = ' + str(self.p) + ', d = ' + str(self.d) + ', q = ' + str(self.q) + ', intercept = ' + str(self.intercept) + ', phi = ' + str(self.phi) + ', theta = ' + str(self.theta) +')'
        

    def params2vector(self):
        """ Parameters to vector
        
        Args:
            None.
            
        Returns:
            

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
        
        if lon_ts <= self.d:
            diff = 0
        else:
            diff = ts[-1] - ts[-self.d]
            
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

        result = -diff + self.intercept + p_sum + q_sum

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