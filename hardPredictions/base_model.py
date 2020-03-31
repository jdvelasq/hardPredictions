"""
Base Model
Base structure for creation of new models

Methods:
    calc_error: Estimates error according to SciKit's regression metrics
    filter_ts: Returns model's residuals

"""

from hardPredictions import series_viewer

import pandas
import numpy
import sklearn
import matplotlib
from extras import add_next_date

class base_model():
    
    def __init__(self):
        self.residuals = None        
        self.test()
        
    def test(self):
        """ Raises error if there are not any of the necessary methods defined """
        
        if (not "predict" in dir(self)):
            raise ValueError('Method "predict" has not been defined')
        if (not "fit" in dir(self)):
            raise ValueError('Method "fit" has not been defined')
        if (not "forecast" in dir(self)):
            raise ValueError('Method "forecast" has not been defined')    
        
    
    def calc_error(self, ts, error_function = None, ignore_first = None):
        """ Estimates error according to SciKit's regression metrics
        
        Args:
            ts: Time series to estimate the model
            error_function (None or error function): Error function whose
                parameters are real time series and estimated time series. If
                None, error_function is Sci-Kit learn's mean squared error
        
        """
        if ignore_first != None:
            ignore = ignore_first
        else:
            try:
                ignore = self.p
            except:
                ignore = 0
                
        y_estimated = self.predict(ts)[ignore:]
        y_real = ts[ignore:]
        
        if (error_function == None):
            error = sklearn.metrics.mean_squared_error(y_real, y_estimated)
        else:
            error = error_function(y_real, y_estimated)      
        
        return error

    
    def filter_ts(self, ts, ignore_first = 0):
        """ Returns model's residuals
        
        Args:
            ts: Time series to estimate residuals
            
        """
        prediction = self.predict(ts)[ignore_first:]
        residuals = ts[ignore_first:].subtract(prediction)
        return residuals            
   
    
    def set_residuals(self, residuals):
        self.residuals = series_viewer(residuals)    
        
    
    """ Residuals analysis """
    def time_plot(self):
        self.residuals.time_plot()
        
    def ACF_plot(self):
        self.residuals.ACF_plot()
    
    def PACF_plot(self):
        self.residuals.PACF_plot()
        
    def qq_plot(self):
        self.residuals.qq_plot()
        
    def density_plot(self):
        self.residuals.density_plot()
        
    def histogram(self):
        self.residuals.histogram()
    
    def normality(self):
        self.residuals.normality()
        
    def simulate(self, ts, periods = 5, confidence_interval = 0.95, iterations = 500):
        values = self.filter_ts(ts, self.p).values
        results = list()
        for i in range(iterations):

            for j in range(periods):
                train = sklearn.utils.resample(values, n_samples = 1)

                if j == 0:
                    y = ts
                else:
                    y = add_next_date(y, next_value_bootstrap)

                next_value = self.forecast(y, 1).series
                next_value_bootstrap = next_value + train[0]
                result_complete = add_next_date(y, next_value_bootstrap)
                result = result_complete[-periods:]

            results.append(result)

        results = pandas.DataFrame(results)
        ci_inf = results.quantile(1-confidence_interval)
        ci_sup = results.quantile(confidence_interval)
        ci = pandas.DataFrame([ci_inf, ci_sup], index = ['ci_inf', 'ci_sup'])

        return ci
    
    def simulate_bootstrap(self, ts, ts_test, confidence_interval = 0.95, iterations = 300):
        periods = len(ts_test)
        values = self.filter_ts(ts, self.p).values
        #results = list()
        errors = list()
        for i in range(iterations):

            for j in range(periods):
                train = sklearn.utils.resample(values, n_samples = 1)

                if j == 0:
                    y = ts
                else:
                    y = add_next_date(y, next_value_bootstrap)

                next_value = self.forecast(y, 1).series
                next_value_bootstrap = next_value + train[0]
                result_complete = add_next_date(y, next_value_bootstrap)
                result = result_complete[-periods:]
            
            error = sklearn.metrics.mean_squared_error(ts_test, result)

            #results.append(result)
            errors.append(error)
            
        mean_error = numpy.mean(errors)      
        

        #results = pandas.DataFrame(results)
        #ci_inf = results.quantile(1-confidence_interval)
        #ci_sup = results.quantile(confidence_interval)
        #ci = pandas.DataFrame([ci_inf, ci_sup], index = ['ci_inf', 'ci_sup'])

        return mean_error


    def plot(self, ts, periods = 5, confidence_interval = None, iterations = 300, ignore_first = None):
        
        fitted_ts = self.predict(ts)
        fitted_ts.index = ts.index
        last = ts[-1:]
        
        if ignore_first != None:
            ignore = ignore_first
        else:
            try:
                ignore = self.p
            except:
                ignore = 0
        
        fitted_ts_plot = fitted_ts[ignore:]
        
        if periods == False:
            pass
        else:
            forecast_ts = self.forecast(ts, periods, confidence_interval, iterations)
            ci_inf = last.append(forecast_ts['ci_inf'])
            ci_sup = last.append(forecast_ts['ci_sup'])
            tseries = last.append(forecast_ts['series'])
        
        if periods == False:
            matplotlib.pyplot.plot(ts, 'k-')
            matplotlib.pyplot.plot(fitted_ts_plot, 'b-')
            matplotlib.pyplot.legend(['Real', 'Fitted'])
        else:
            matplotlib.pyplot.plot(ts, 'k-')  
            matplotlib.pyplot.plot(fitted_ts_plot, 'c-')                      
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
        tscv = sklearn.model_selection.TimeSeriesSplit(n_splits = n_splits)
        splits = tscv.split(X)

        error_list = list()
        for train_index, test_index in splits:
            y_train, y_test = y[train_index], y[test_index]
            y_train_index, y_test_index = y_index[train_index], y_index[test_index]

            y_train = pandas.Series((v for v in y_train), index = y_train_index)
            y_test = pandas.Series((v for v in y_test), index = y_test_index)
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
                train = sklearn.utils.resample(values, n_samples = 1)
                new_value = train[0] + serie[j]
                result.append(new_value)

            results.append(result)

        results = pandas.DataFrame(results)
        minim = results.quantile(1-confidence_interval)
        maxim = results.quantile(confidence_interval)
        final_result = pandas.DataFrame([minim, maxim])

        return final_result
