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

from base_model import base_model

from AR import AR

import numpy
import scipy
import pandas
import matplotlib
from extras import add_next_date
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample

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