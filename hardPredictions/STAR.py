"""

STAR Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains STAR models

Examples
-------------------------------------------------------------------------------


Classes
-------------------------------------------------------------------------------

"""

from base_model import base_model

import numpy
import scipy
import pandas
import matplotlib
import math
from extras import add_next_date
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample

class STAR(base_model):
    """ 

    """

    def __init__(self, p=None, gamma=None, center=None, intercept_1=None, intercept_2 =None, phi_1=None, phi_2=None):
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p
            
        
        self.gamma = gamma
        self.center = center
        self.intercept_1 = intercept_1
        self.intercept_2 = intercept_2
        self.phi_1 = phi_1
        self.phi_2 = phi_2          
        
        self.gamma_value = (gamma == None)
        self.center_value = (center == None)
        self.intercept_1_value = (intercept_1 == None)
        self.phi_1_value = (phi_1 == None)
        self.intercept_2_value = (intercept_2 == None)
        self.phi_2_value = (phi_2 == None)
        
        self.optim_type = 'complete'
        
            
        
    def __repr__(self):
        return 'STAR(p = ' + str(self.p) + ', gamma = ' + str(self.gamma) + ', center = ' + str(self.center) +  ', intercept_1 = ' + str(self.intercept_1) +  ', phi_1 = ' + str(self.phi_1) +  ', intercept_2 = ' + str(self.intercept_2) + ', phi_2 = ' + str(self.phi_2) +')'
        

    def params2vector(self):
        """ Parameters to vector
        
        

        """        
        params = list()
        if self.gamma == None:
            self.gamma = numpy.random.rand(1)[0]
        if self.center == None:
            self.center = numpy.random.rand(1)[0]    
        if self.intercept_1 == None:
            self.intercept_1 = numpy.random.rand(1)[0]
        if self.phi_1 == None:
            self.phi_1 = numpy.random.rand(self.p)
        if self.intercept_2 == None:
            self.intercept_2 = numpy.random.rand(1)[0]
        if self.phi_2 == None:
            self.phi_2 = numpy.random.rand(self.p)
        
        if self.optim_type == 'complete':
            if self.gamma_value:
                params.append(self.gamma)
            if self.center_value:
                params.append(self.center)
            if self.intercept_1_value:
                params.append(self.intercept_1)
            if self.phi_1_value:
                for i in range(len(self.phi_1)):
                    params.append(self.phi_1[i])
            if self.intercept_2_value:
                params.append(self.intercept_2)
            if self.phi_2_value:
                for i in range(len(self.phi_2)):
                    params.append(self.phi_2[i])
            return params
        elif self.optim_type == 'no_optim':
            pass
        

    def vector2params(self, vector):
        """ Vector to parameters
        

        """ 
        
        if self.optim_type == 'complete':
            i = 0
            if self.gamma_value:
                self.gamma = vector[i]
                i = i+1
            if self.center_value:
                self.center = vector[i]
                i = i+1
            if self.intercept_1_value:
                self.intercept_1 = vector[i]
                i = i+1
            if self.phi_1_value:
                self.phi_1 = vector[i:i+self.p]
                i = i+1
            if self.intercept_2_value:
                self.intercept_2 = vector[3+self.p]
                i = i+1
            if self.phi_2_value:
                self.phi_2 = vector[4+self.p:4+2*self.p]  
        elif self.optim_type == 'no_optim':
            pass
            
        return self
    
    def __forward__(self, y):
        y = y.values
        lon = len(y)
        if lon <= self.p:
            z_t = numpy.mean(y)
            G_zt = 1/(1+math.exp(-1*self.gamma*(z_t-self.center)))
            y_last = y[0:lon]
            result = self.intercept_1*G_zt + numpy.dot(y_last, self.phi_1[0:lon])*G_zt + self.intercept_2*(1-G_zt) + numpy.dot(y_last, self.phi_2[0:lon])*(1-G_zt)
        else:
            z_t = y[-(self.p+1)]
            G_zt = 1/(1+math.exp(-1*self.gamma*(z_t-self.center)))
            y_last = y[lon-self.p:lon]
            result = self.intercept_1*G_zt + numpy.dot(y_last, self.phi_1)*G_zt + self.intercept_2*(1-G_zt) + numpy.dot(y_last, self.phi_2)*(1-G_zt)

        return result

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
            def f(x):
                self.vector2params(x)
                return self.calc_error(ts, error_function, self.p)

            x0 = self.params2vector()
            optim_params = scipy.optimize.minimize(f, x0)
            self.vector2params(vector = optim_params.x)

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