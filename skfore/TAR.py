"""

TAR Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains TAR models using SciPy's minimization method for AR based 
models.

Examples
-------------------------------------------------------------------------------



"""

from skfore.base_model import base_model

from AR import AR

import numpy
import scipy
import pandas
from skfore.extras import add_next_date

class TAR(base_model):
    """ 

    """

    def __init__(self, p=None, intercept_1=None, intercept_2 =None, phi_1=None, phi_2=None, max_interval = 10):
        
        
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p
        
        self.intercept_1 = intercept_1
        self.intercept_2 = intercept_2
        self.phi_1 = phi_1
        self.phi_2 = phi_2        
        
        self.max_interval = max_interval
        self.d = max_interval
        self.ts_1 = None
        self.ts_2 = None

            
        
    def __repr__(self):
        return 'TAR(p = ' + str(self.p) + ', intercept_1 = ' + str(self.model_1.phi0) + ', intercept_2 = ' + str(self.model_2.phi0) + ', phi_1 = ' + str(self.model_1.phi) + ', phi_2 = ' + str(self.model_2.phi) +')'
        

    def predict(self, ts):
        """ Fits a time series using self model parameters
        
        Args:
            ts (pandas.Series): Time series to fit.
        
        Returns:
            Fitted time series.
            
        """

        prediction_1 = self.model_1.predict(self.ts_1)
        prediction_2 = self.model_2.predict(self.ts_2)
        prediction = prediction_1.append(prediction_2)
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
        errors = list()
        tes = list()
        for d in range(self.max_interval, len(ts)-self.max_interval):
            self.ts_1 = ts[0:d]
            self.ts_2 = ts[d:]
            self.model_1 = AR(p = self.p, intercept = self.intercept_1, phi = self.phi_1)
            self.model_1 = self.model_1.fit(self.ts_1, error_function)
            self.model_2 = AR(p = self.p, intercept = self.intercept_2, phi = self.phi_2)
            self.model_2 = self.model_2.fit(self.ts_2, error_function)
            error = self.calc_error(ts)
            errors.append(error)            
            tes.append(d)
        minim = errors.index(min(errors))
        self.d = tes[minim]
        self.ts_1 = ts[0:self.d]
        self.ts_2 = ts[self.d:]
        self.model_1 = AR(p = self.p, intercept = self.intercept_1, phi = self.phi_1)
        self.model_1 = self.model_1.fit(self.ts_1, error_function)
        self.model_2 = AR(p = self.p, intercept = self.intercept_2, phi = self.phi_2)
        self.model_2 = self.model_2.fit(self.ts_2, error_function)

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
                y = ts[self.d:]

            value = self.model_2.__forward__(y)
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