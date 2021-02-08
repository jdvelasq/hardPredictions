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

Get predicted values as a DataFrame:
   
Load time series
>>> ts = load_champagne()

>>> random.seed(100)
>>> model = AMNFIS(p = 4, c = 4)
>>> model  # doctest: +ELLIPSIS
AMNFIS(p = 4, c = 4, phi0 = [0.504... 0.610... 0.129... 0.5783...], phi = [[0.847... 0.517... 0.656... 0.978...]
 [0.105...  0.830... 0.595... 0.047...]
 [0.094... 0.997... 0.106... 0.893...]
 [0.121... 0.189... 0.833... 0.415...]])

>>> random.seed(100)
>>> model.fit(ts) # doctest: +ELLIPSIS
AMNFIS(p = 4, c = 4, phi0 = [112994..., -57501..., -58520..., 83910...], phi = [[  -2...   -1...  -12...  -30...]
 [ 935... -556...  15...  954...]
 [-926...  554...  -14... -953...]
 [ -12...    2...    10...   34...]])

>>> random.seed(100)
>>> model.predict(ts, periods = 2) # doctest: +ELLIPSIS
                 ci_inf        ci_sup  ...      forecast  real
1972-10-01  4668...   8992...  ...   6700...  None
1972-11-01  5502...  11967...  ...   8616...  None
<BLANKLINE>
[2 rows x 6 columns]

Classes
-------------------------------------------------------------------------------

"""

from base_model import base_model
from datasets import *

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
            self.p = 0
        else:
            self.p = p
            
        if c == None:
            self.c = 1
        else:
            self.c = c   
            
        self.phi0 = phi0
        self.phi = phi  
        self.centers = None
        self.w = list()
        
        if self.phi0 == None:
            self.phi0 = numpy.random.rand(self.c)
        if self.phi == None:
            self.phi = numpy.random.rand(self.c, self.p)
            
        self.optim_type = optim_type
            
        
    def __repr__(self):
        return 'AMNFIS(p = ' + str(self.p) + ', c = ' + str(self.c) + ', phi0 = ' + str(self.phi0) +', phi = ' + str(self.phi) + ')'
        

    def params2vector(self):
        """ Parameters to vector
        
        Args:
            None
            
        Returns:
            Vector parameters of length (c+1)*p to use in optimization
           

        """        
        params = list()        
        if self.optim_type == 'complete':            
            for i in range(self.c):
                params.append(self.phi0[i])
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
            new_vector = numpy.reshape(vector, (self.c, self.p+1))
            phi0 = list()
            for i in range(self.c):
                phi0.append(new_vector[i,0])
            self.phi0 = phi0
            self.phi = new_vector[:,1:]
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
        

    def forecast(self, ts):
        """ Next step 
        
        Args:
            ts (pandas.Series): Time series to find next value
            
        Returns:
            Value of next time stamp
            
        """
        
        y = ts.values
        lon = len(y)
        y_last = y[lon-self.p:lon]        
        wi = self.__calc_wi__(y_last)
        p = list()
            
        for i in range(self.c):            
            result = (self.phi0[i] + numpy.dot(y_last, self.phi[i]))*wi[i]
            p.append(result)
        
        total = sum(p)            

        return total

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
            if i < self.p:
                result = 0
            else:
                result = self.forecast(y[0:i])
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
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.ELLIPSIS)