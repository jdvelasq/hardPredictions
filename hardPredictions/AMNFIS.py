"""

Adaptive Multidimensional Neuro-Fuzzy Inference System
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
import matplotlib
import random
import math
from extras import add_next_date
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample

class AMNFIS(base_model):
    """ AMNFIS model

    Args:
        c (int): number of centers.
        p (int): order of AR models.

    Returns:
        AMNFIS structure with c AR models of order p.

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
            None.
            
        Returns:
           

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
            ts (pandas.Series): Time series to fit.
        
        Returns:
            Fitted time series.
            
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
            ts (pandas.Series): Time series to fit.
            error_function (function): Function to estimates error.
            
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
    
    def simulate(self, ts, periods = 5, confidence_interval = 0.95, iterations = 1000):
        values = self.filter_ts(ts, self.p).values
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
        fitted_ts = fitted_ts[self.p:]
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