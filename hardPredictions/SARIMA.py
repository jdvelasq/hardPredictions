"""

SARIMA Model
===============================================================================

Overview
-------------------------------------------------------------------------------



Classes
-------------------------------------------------------------------------------

"""

from base_model import base_model

import numpy
import scipy
import pandas
import statsmodels
from extras import add_next_date


class SARIMA(base_model):
    """ Autoregressive model
    
    Parameter optimization method: scipy's minimization

    Args:
        p (int): order
        intercept (boolean or double): False for set intercept to 0 or double 
        phi (array): array of p-length for set parameters without optimization

    Returns:
        AR model structure of order p

    """

    def __init__(self, p=None, d=None, q=None, P=None, D=None, Q=None, m=None, **kwargs):
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
        
        if P == None:
            raise ValueError('Please insert parameter P')
        else:
            self.P = P
        
        if D == None:
            raise ValueError('Please insert parameter D')
        else:
            self.D = D
        
        if Q == None:
            raise ValueError('Please insert parameter Q')
        else:
            self.Q = Q
        
        if m == None:
            self.m = 0
        else:
            self.m = m
            
        self.model = None     
        
            
        
    def __repr__(self):
        return 'SARIMA(p = ' + str(self.p) +')'
        
    
    def __forward__(self, y):
        
        result = self.model.predict(start=len(y), end=len(y))
        
        return result.values[0]

    def predict(self, ts):
        """ Fits a time series using self model parameters
        
        Args:
            ts (pandas.Series): Time series to fit
    
        Returns:
            Fitted time series
            
        """
        y = ts
        prediction = list()
        for i in range(len(y)):
            result = self.__forward__(y[0:i])
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
        
        self.model = statsmodels.tsa.statespace.sarimax.SARIMAX(ts, order = (self.p,self.d,self.q), seasonal_order = (self.P,self.D,self.Q,self.m), **kwargs)
        self.model = self.model.fit()
        
        return self
    

    def forecast(self, ts, periods, confidence_interval = None, iterations = 300):
        """ Predicts future values in a given period
        
        Args:
            ts (pandas.Series): Time series to predict
            periods (int): Number of periods ahead to predict
            confidence_interval (double): Confidence interval level
            iterations (int): Number of iterations
            
        Returns:
            Dataframe of confidence intervals and time series of predicted 
            values: (ci_inf, ci_sup, series) 
        
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