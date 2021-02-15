"""
Base Model
Base structure for creation of new models

Methods:
    calc_error: Estimates error according to SciKit's regression metrics
    filter_ts: Returns model's residuals

"""

import sys
sys.path.append('../')

from skfore.skfore import series_viewer
from skfore.datasets import *

import pandas
import numpy
import scipy
import sklearn
import matplotlib
import random
import math
from skfore.extras import add_next_date
from sklearn import preprocessing

class base_model():

    def __init__(self):
        self.residuals = None
        self.scaler = None
        self.test()

    def test(self):
        """ Raises error if there are not any of the necessary methods defined """

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
                ignore = self.q
            except:
                try:
                    ignore = self.p
                except:
                    ignore = 0

        y_estimated = self.simulate(ts)[ignore:]
        y_real = ts[ignore:]

        if (error_function == None):
            error = sklearn.metrics.mean_squared_error(y_real, y_estimated)
        else:
            error = error_function(y_real, y_estimated)

        return error


    def filter_ts(self, ts, ignore_first = None):
        """ Returns model's residuals

        Args:
            ts: Time series to estimate residuals

        """

        if ignore_first != None:
            ignore = ignore_first
        else:
            try:
                ignore = self.q
            except:
                try:
                    ignore = self.p
                except:
                    ignore = 0

        prediction = self.simulate(ts)[ignore:]
        residuals = ts[ignore:].subtract(prediction)
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


    def update(self, kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        return self


    def predict(self, ts, periods, tsp = None, blind = True, confidence_interval = None, iterations = 300, error_sample = 'bootstrap', ignore_first = None, random_state = 100):
        """ Predicts future values in a given period

        Args:
            ts (pandas.Series): Time series to predict
            periods (int): Number of periods ahead to predict
            tsp (pandas.Series): Predicted time series to compare future values
            blind (boolean): True to forecast without using predicted time
                series or False to use it in forecasting
            confidence_interval (double): Confidence interval level
            iterations (int): Number of iterations
            error_sample (str): Use 'bootstrap' to forecast using sample errors
                of filtered time series or 'normal' to forecast using errors
                from a gaussian distribution with known variance
            random_state (int): Determines random number generation for seed

        Returns:
            Dataframe of confidence intervals and time series of predicted
            values: (ci_inf, ci_sup, series)

        """

        random.seed(random_state)



        if blind == False:
            if tsp is None:
                raise ValueError('Predicted time series not defined for no blind forecast')
            else:
                if error_sample == 'bootstrap':

                    if confidence_interval is None:
                        c_i = 0.95
                    else:
                        c_i = confidence_interval

                    for i in range(len(tsp)):
                        if i == 0:
                            tse = ts
                            simul_step = self.bootstrap(tse, 1, confidence_interval = c_i, iterations = iterations)
                            simul_result = simul_step.transpose()
                            y = ts
                        else:
                            tse = ts.append(tsp[0:i])
                            simul_step = self.bootstrap(tse, 1, confidence_interval = c_i, iterations = iterations)
                            simul_result = simul_result.append(simul_step.transpose())

                        value = self.forecast(y)
                        y = add_next_date(y, value)

                    prediction = y[-len(tsp):]
                    prediction.name = 'series'
                    ci = pandas.DataFrame([prediction], index = ['series'])

                    result = ci.append(simul_result.transpose())


                elif error_sample == 'normal':
                    if confidence_interval is None:
                        for i in range(len(tsp)):
                            if i == 0:
                                tse = ts
                                simul_step = self.normal_error(1, tse, ignore_first)
                                simul_result = simul_step
                                y = ts
                            else:
                                tse = ts.append(tsp[0:i])
                                simul_step = self.normal_error(1, tse, ignore_first)
                                simul_result = simul_result.append(simul_step)

                            value = self.forecast(y)
                            y = add_next_date(y, value)

                        prediction = y[-len(tsp):]
                        prediction.name = 'series'
                        ci = pandas.DataFrame([prediction], index = ['series'])

                        result = ci.append(simul_result.transpose())


                    else:
                        for i in range(len(tsp)):
                            if i == 0:
                                tse = ts
                                simul_step = self.normal_error(1, tse, ignore_first)
                                simul_step_b = self.bootstrap(tse, 1, confidence_interval = confidence_interval, iterations = iterations)
                                simul_result = simul_step
                                simul_result_b = simul_step_b.transpose()
                                y = ts
                            else:
                                tse = ts.append(tsp[0:i])
                                simul_step = self.normal_error(1, tse, ignore_first)
                                simul_step_b = self.bootstrap(tse, 1, confidence_interval = confidence_interval, iterations = iterations)
                                simul_result = simul_result.append(simul_step)
                                simul_result_b = simul_result_b.append(simul_step_b.transpose())

                            value = self.forecast(y)
                            y = add_next_date(y, value)

                        prediction = y[-len(tsp):]
                        prediction.name = 'series'
                        ci = pandas.DataFrame([prediction], index = ['series'])

                        result = ci.append(simul_result.transpose())
                        result = result.append(simul_result_b.transpose())


                else:
                    raise ValueError('Error sample has not been defined correctly')
        else:
            for i in range(periods):
                if i == 0:
                    y = ts
                value = self.forecast(y)
                y = add_next_date(y, value)

            if error_sample == 'bootstrap':
                if confidence_interval is None:
                    c_i = 0.95
                else:
                    c_i = confidence_interval

                ci = self.bootstrap(ts, periods, c_i, iterations)
                prediction = y[-periods:]
                prediction.name = 'series'
                result = ci.append(prediction)

            elif error_sample == 'normal':
                prediction = y[-periods:]
                prediction.name = 'series'
                ci = pandas.DataFrame([prediction], index = ['series'])

                if confidence_interval is None:

                    simulation = self.normal_error(periods, ts, ignore_first)
                    result = ci.append(simulation.transpose())

                else:

                    simulation = self.normal_error(periods, ts, ignore_first)
                    simulation_b = self.bootstrap(ts, periods, confidence_interval, iterations)
                    result = ci.append(simulation.transpose())
                    result = result.append(simulation_b)

            else:
                raise ValueError('Error sample has not been defined correctly')

        result = result.transpose()
        if error_sample == 'bootstrap':
            result['forecast'] = result.bootstrap
        elif error_sample == 'normal':
            result['forecast'] = result.normal

        result['real'] = tsp

        return result


    def bootstrap(self, ts, periods = 5,confidence_interval = 0.95, iterations = 500):

        try:
            ignore = self.q
        except:
            try:
                ignore = self.p
            except:
                ignore = 0
        values = self.filter_ts(ts, ignore).values
        results = list()
        for i in range(iterations):

            for j in range(periods):
                train = sklearn.utils.resample(values, n_samples = 1)

                if j == 0:
                    y = ts
                else:
                    y = add_next_date(y, next_value_bootstrap)

                next_value = self.forecast(y)
                next_value_bootstrap = next_value + train[0]
                result_complete = add_next_date(y, next_value_bootstrap)
                result = result_complete[-periods:]

            results.append(result)

        results = pandas.DataFrame(results)
        ci_inf = results.quantile(1-confidence_interval)
        ci_sup = results.quantile(confidence_interval)
        mean = results.mean()
        ci = pandas.DataFrame([ci_inf, ci_sup, mean], index = ['ci_inf', 'ci_sup', 'bootstrap'])

        return ci

    def normal_error(self, n, ts, ignore_first = None):
        residuals = self.filter_ts(ts, ignore_first)
        var = numpy.std(residuals)
        generated_values = numpy.random.normal(0, var, n)

        for i in range(n):
            if i == 0:
                y = ts
            value = self.forecast(y)
            value = value + generated_values[i]
            y = add_next_date(y, value)

        result = pandas.DataFrame(y[-n:].values, index = y[-n:].index, columns = ['normal'])

        return result

    def plot_forecast(self, ts, periods = 5, tsp = None, blind = True, confidence_interval = None, iterations = 300, ignore_first = None):

        fitted_ts = self.simulate(ts)
        fitted_ts.index = ts.index
        last = ts[-1:]

        if ignore_first != None:
            ignore = ignore_first
        else:
            try:
                ignore = self.q
            except:
                try:
                    ignore = self.p
                except:
                    ignore = 0

        fitted_ts_plot = fitted_ts[ignore:]

        if periods == False:
            pass
        else:
            forecast_ts = self.predict(ts, periods, tsp, blind, confidence_interval, iterations)
            ci_inf = last.append(forecast_ts['ci_inf'])
            ci_sup = last.append(forecast_ts['ci_sup'])
            tseries = last.append(forecast_ts['forecast'])

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

            if tsp is None:
                pass
            else:
                matplotlib.pyplot.plot(tsp, 'k-')

            if confidence_interval != None:
                matplotlib.pyplot.legend(['Real', 'Fitted', 'Forecast', 'CI', 'CI'])
            else:
                matplotlib.pyplot.legend(['Real', 'Fitted', 'Forecast'])

    def plot(self, ts, forecast = None, ignore_first = None):

        fitted_ts = self.simulate(ts)
        fitted_ts.index = ts.index
        last = ts[-1:]

        if ignore_first != None:
            ignore = ignore_first
        else:
            try:
                ignore = self.p
            except:
                try:
                    ignore = self.q
                except:
                    ignore = 0

        fitted_ts_plot = fitted_ts[ignore:]

        if forecast is None:
            matplotlib.pyplot.plot(ts, 'k-')
            matplotlib.pyplot.plot(fitted_ts_plot, 'b-')
            matplotlib.pyplot.legend(['Real', 'Fitted'])
            matplotlib.pyplot.legend(['Real', 'Fitted', 'Forecast'])
        else:
            matplotlib.pyplot.plot(ts, 'k-')
            matplotlib.pyplot.plot(fitted_ts_plot, 'c-')
            tseries = last.append(forecast['forecast'])
            matplotlib.pyplot.plot(tseries, 'b-')

            if 'ci_inf' in forecast:
                ci_inf = last.append(forecast['ci_inf'])
                matplotlib.pyplot.plot(ci_inf, 'r--')
            if 'ci_sup' in forecast:
                ci_sup = last.append(forecast['ci_sup'])
                matplotlib.pyplot.plot(ci_sup, 'r--')

            matplotlib.pyplot.plot(forecast['real'], 'k-')
            matplotlib.pyplot.axvline(x = ts[-1:].index, color = 'k', linestyle = '--')

            if 'ci_inf' in forecast:
                matplotlib.pyplot.legend(['Real', 'Fitted', 'Forecast', 'CI', 'CI'])


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


    #def get_predict_ci(self, ts, confidence_interval = 0.95, iterations = 1000):
    #    values = self.filter_ts(ts).values
    #    serie = self.simulate(ts).values
    #    results = list()
    #    for i in range(iterations):
    #        result = list()
    #        for j in range(len(serie)):
    #            train = sklearn.utils.resample(values, n_samples = 1)
    #            new_value = train[0] + serie[j]
    #            result.append(new_value)

    #        results.append(result)

    #    results = pandas.DataFrame(results)
    #    minim = results.quantile(1-confidence_interval)
    #    maxim = results.quantile(confidence_interval)
    #    final_result = pandas.DataFrame([minim, maxim])

    #    return final_result

    def normalize(self, ts):

        if self.scaler == None:
            scaler = preprocessing.MinMaxScaler()
            values = ts.values.reshape((len(ts.values), 1))
            scaler.fit(values)
            self.scaler = scaler
        else:
            values = ts.values.reshape((len(ts.values), 1))
            scaler = self.scaler
        normalized = scaler.transform(values)
        norm_series = pandas.Series((v[0] for v in normalized), index = ts.index)

        return norm_series

    def des_normalize(self, ts):
        values = ts.values.reshape((len(ts.values), 1))
        des_norm = self.scaler.inverse_transform(values)
        des_norm_series = pandas.Series((v[0] for v in des_norm), index = ts.index)
        return des_norm_series
