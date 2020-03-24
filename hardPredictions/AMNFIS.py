"""

Adaptive Multidimensional Neuro-Fuzzy Inference System
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains an adaptive multidimensional neuro-fuzzy inference system
(AMNFIS), developed originally for processes control. This model was taken from
paper "Adaptive Multidimensional Neuro-Fuzzy  Inference System for Time Series 
Prediction" (J.D. Velásquez) in IEEE LATIN AMERICA TRANSACTIONS, VOL. 13, 
NO. 8, AUG. 2015.


Examples
-------------------------------------------------------------------------------

>>> ts = pandas.Series.from_csv('../datasets/WWWusage.csv', index_col = 0, header = 0)
>>> model = AMNFIS(p = 4, c = 4)
>>> model.fit(ts)
AMNFIS(p = 4, c = 4)
>>> 

Classes
-------------------------------------------------------------------------------

"""

from base_model import base_model

import numpy
import scipy
import pandas
import random
import math
from sklearn import *
from extras import add_next_date

class AMNFIS(base_model):
    """ AMNFIS model

    Args:
        p (int): order of AR models
        c (int): number of centers
        optim_type (str): character string of 'complete' if all paremeters are 
        wanted or or 'no_optim' to take given parameters
        phi0 (None or double): None to find it or double to set it
        phi (None or array): array of p-length for set parameters without 
        optimization

    Returns:
        AMNFIS structure with c AR models of order p

    """

    def __init__(self, p=None, c=None, optim_type='complete', phi0=None, phi=None):
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p
            
        if c == None:
            raise ValueError('Please insert parameter c')
        else:
            self.c = c   
            
        self.phi0 = phi0
        self.phi = phi  
        self.centers = None
        self.w = list()
        
        if self.phi0 == None:
            self.phi0 = numpy.random.rand(self.p)
        if self.phi == None:
            self.phi = numpy.random.rand(self.c, self.p)
            
        self.optim_type = optim_type
            
        
    def __repr__(self):
        return 'AMNFIS(p = ' + str(self.p) + ', c = ' + str(self.c) +')'
        

    def params2vector(self):
        """ Parameters to vector
        
        Args:
            None
            
        Returns:
            Vector parameters of length (c+1)*p to use in optimization
           

        """        
        params = list()        
        if self.optim_type == 'complete':
            params.append(self.phi0)
            for i in range(self.c):
                params.append(self.phi[i])
            result = numpy.hstack(params)
            return result
        elif self.optim_type == 'no_optim':
            pass
        

    def vector2params(self, vector):
        """ Vector to parameters
        
        Args:
            vector (array): Vector parameters of length (c+1)*p to convert into 
            parameters of the model
            
        Returns:
            self

        """ 
        
        if self.optim_type == 'complete':
            new_vector = numpy.reshape(vector, (self.c+1, self.p))
            self.phi0 = new_vector[0]
            self.phi = new_vector[1:]
        elif self.optim_type == 'no_optim':
            pass
            
        return self
    
    def __select_centers__(self, ts):
        self.centers = list()
        for i in range(self.c):
            r = random.randrange(len(ts))
            self.centers.append(ts.values[r])
        
        return self

    def __calc_miu__(self, vector):
        D2 = list()
        for i in range(self.c):
            di = list()
            for j in vector:
                dii = (j-self.centers[i])**2
                di.append(dii)                
            
            Di2 = (math.sqrt(sum(di)))**2            
            D2.append(Di2)
            
        miui = [math.exp(item/sum(D2)) for item in D2]
         
        return miui
    
    def __calc_wi__(self, vector):
        miu = self.__calc_miu__(vector)
        wi = [item/sum(miu) for item in miu]
        
        return wi
        

    def __forward__(self, y):
        y = y.values
        lon = len(y)
        y_last = y[lon-self.p:lon]        
        wi = self.__calc_wi__(y_last)
        p = list()
            
        for i in range(self.c):            
            result = (self.phi0[i] + numpy.dot(y_last, self.phi[i]))*wi[i]
            p.append(result)
        
        total = sum(p)            

        return total

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
            if i < self.p:
                result = 0
            else:
                result = self.__forward__(y[0:i])
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
            if self.centers == None:
                self.centers = self.__select_centers__(ts).centers
                
            def f(x):
                self.vector2params(x)
                return self.calc_error(ts, error_function)

            x0 = self.params2vector()
            optim_params = scipy.optimize.minimize(f, x0)
            self.vector2params(vector = optim_params.x)
            
            def fc(x):
                self.centers = x
                return self.calc_error(ts, error_function)
            
            xc0 = self.centers
            optim_centers = scipy.optimize.minimize(fc, xc0)
            self.centers = optim_centers.x

        return self

    def forecast(self, ts, periods, confidence_interval = None, iterations = 300):
        """ Predicts future values in a given period
        
        Args:
            ts (pandas.Series): Time series to predict.
            periods (int): Number of periods ahead to predict.
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
