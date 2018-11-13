"""

Holt Winters Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains Holt Winters or Exponential Smoothing model.


Examples
-------------------------------------------------------------------------------

All parameters can be optimized by choosing seasonal type: additive or
multiplicative. Additive seasonal is set by default.

>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = HoltWinters()
>>> model
HoltWinters(alpha = None, beta = None, gamma = None, seasonal = additive)
>>> model = model.fit(ts)
>>> model
HoltWinters(alpha = 0.08779225079496673, beta = -0.03280832897112478, gamma = 0.9985667470201687, seasonal = additive)
>>> fitted_model = model.predict(ts)
>>> prediction = model.forecast(ts, periods = 2)
>>> prediction
            ci_inf  ci_sup       series
1972-10-01     NaN     NaN  6841.767847
1972-11-01     NaN     NaN  9754.197706
>>> prediction = model.forecast(ts, periods = 2, confidence_interval = 0.95)
>>> prediction
                 ci_inf       ci_sup       series
1972-10-01  6817.142357  6851.141877  6841.767847
1972-11-01  9729.312585  9763.392881  9754.197706
>>> model.plot(ts, periods = 2, confidence_interval = 0.95)

.. image:: ./images/HW.png
  :width: 400
  :alt: AR 1
  :align: center

None parameters will be optimized even if other parameters are set:

>>> model = HoltWinters(alpha = 0.9)
>>> model
HoltWinters(alpha = 0.9, beta = None, gamma = None, seasonal = additive)
>>> model = model.fit(ts)
>>> model
HoltWinters(alpha = 0.9, beta = 0.04986683901737994, gamma = 0.2825125430030021, seasonal = additive)
>>> prediction = model.forecast(ts, periods = 2, confidence_interval = 0.95)
>>> prediction
                 ci_inf       ci_sup       series
1972-10-01  6903.954800  7075.239583  7012.853656
1972-11-01  9648.596015  9897.577641  9836.879742
>>> model.plot(ts, periods = 2, confidence_interval = 0.95)

.. image:: ./images/HW_alpha.png
  :width: 400
  :alt: AR 1
  :align: center

Parameters can also be False if they do not want to be found:

>>> model = HoltWinters(alpha = 0.9, beta = 0.1, gamma = False)
>>> model
HoltWinters(alpha = 0.9, beta = 0.1, gamma = False, seasonal = additive)
>>> model = model.fit(ts)
>>> model
HoltWinters(alpha = 0.9, beta = 0.1, gamma = False, seasonal = additive)
>>> prediction = model.forecast(ts, periods = 2, confidence_interval = 0.95)
>>> prediction
                 ci_inf       ci_sup       series
1972-10-01  4814.564979  5812.226356  5495.803368
1972-11-01  4731.113279  6040.021961  5572.626448
>>> model.plot(ts, periods = 2, confidence_interval = 0.95)

.. image:: ./images/HW_False.png
  :width: 400
  :alt: AR 1
  :align: center

"""

from hardPredictions.base_model import base_model

#import numpy
#import scipy
#import pandas
#import matplotlib
from hardPredictions.extras import *
from sklearn.utils import resample

class HoltWinters(base_model):
    """ HW """

    def __init__(self, alpha=None, beta=None, gamma=None, seasonal='additive'):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal = seasonal
        self.fit_model = 'fit'

        """ Checks parameters """
        #if seasonal != 'additive' or seasonal != 'multiplicative':
        #    seasonal_error = 'Error: Invalid seasonal value: ' + self.seasonal
        #    raise ValueError(seasonal_error)

        if (self.alpha != None and self.alpha != False) and (self.beta != None and self.beta != False) and (self.gamma != None and self.gamma != False):
            self.fit_model = 'no fit'
        elif (self.alpha != None and self.alpha != False) and (self.beta != None and self.beta != False) and self.gamma == False:
            self.fit_model = 'no fit'
        elif (self.alpha != None and self.alpha != False) and self.beta == False and self.gamma == False:
            self.fit_model = 'no fit'
        elif (self.alpha != None and self.alpha != False) and self.beta == None and (self.gamma != None and self.gamma != False):
            self.fit_model = 'beta'
        elif (self.alpha != None and self.alpha != False) and self.beta == None and self.gamma != False:
            self.fit_model = 'beta'
        elif (self.alpha != None and self.alpha != False) and self.beta == None and self.gamma == None:
            self.fit_model = 'beta_gamma'
        elif (self.alpha != None and self.alpha != False) and (self.beta != None and self.beta != False) and self.gamma == None:
            self.fit_model = 'gamma'
        elif self.alpha == None and (self.beta != None and self.beta != False) and self.gamma == False:
            self.fit_model = 'alpha'
        elif self.alpha == None and (self.beta != None and self.beta != False) and self.gamma == None:
            self.fit_model = 'alpha_gamma'
        elif self.alpha == None and self.beta == None and (self.gamma != None and self.gamma != False):
            self.fit_model = 'alpha_beta'
        elif self.alpha == None and (self.beta != None and self.beta != False) and (self.gamma != None and self.gamma != False):
            self.fit_model = 'alpha'
        elif self.beta == False and self.gamma == False:
            self.fit_model = 'alpha'
        elif self.gamma == False:
            self.fit_model = 'alpha_beta'
        else:
            self.fit_model = 'alpha_beta_gamma'

        if self.alpha == False:
            message_alpha = 'Error: parameter "alpha" cannot be False'
            raise ValueError(message_alpha)

        if self.alpha != False and self.beta == False and self.gamma != False:
            message_beta = 'Error: parameter "beta" has not been defined'
            raise ValueError(message_beta)

    def __repr__(self):
        return 'HoltWinters(alpha = ' + str(self.alpha) + ', beta = ' + str(self.beta) + ', gamma = ' + str(self.gamma) + ', seasonal = ' + str(self.seasonal) +')'


    def params2vector(self):
        params = list()
        if self.fit_model == 'alpha':
            params.append(self.alpha)
        elif self.fit_model == 'alpha_beta':
            params.append(self.alpha)
            params.append(self.beta)
        elif self.fit_model == 'alpha_beta_gamma':
            params.append(self.alpha)
            params.append(self.beta)
            params.append(self.gamma)
        elif self.fit_model == 'alpha_gamma':
            params.append(self.alpha)
            params.append(self.gamma)
        elif self.fit_model == 'beta_gamma':
            params.append(self.beta)
            params.append(self.gamma)
        elif self.fit_model == 'beta':
            params.append(self.beta)
        elif self.fit_model == 'gamma':
            params.append(self.gamma)

        return params

    def vector2params(self, vector):
        if self.fit_model == 'alpha':
            self.alpha = vector[0]
        elif self.fit_model == 'alpha_beta':
            self.alpha = vector[0]
            self.beta = vector[1]
        elif self.fit_model == 'alpha_beta_gamma':
            self.alpha = vector[0]
            self.beta = vector[1]
            self.gamma = vector[2]
        elif self.fit_model == 'alpha_gamma':
            self.alpha = vector[0]
            self.gamma = vector[1]
        elif self.fit_model == 'beta_gamma':
            self.beta = vector[0]
            self.gamma = vector[1]
        elif self.fit_model == 'beta':
            self.beta = vector[0]
        elif self.fit_model == 'gamma':
            self.gamma = vector[0]

        return self

    def _alpha(self, ts, forecast = False):
        result = [ts[0]]
        if forecast == True:
            x = 1
        else:
            x = 0
        for n in range(1, len(ts) + x):
            if n >= len(ts):
                value = result[-1]
            else:
                value = ts[n]
            result.append(self.alpha * value + (1 - self.alpha) * result[n-1])

        return result

    def _alpha_beta(self, ts, forecast = False):
        result = [ts[0]]
        if forecast == True:
            x = 1
        else:
            x = 0
        for n in range(1, len(ts) + x):
            if n == 1:
                level, trend = ts[0], ts[1] - ts[0]
            if n >= len(ts):
                value = result[-1]
            else:
                value = ts[n]
            last_level, level = level, self.alpha*value + (1-self.alpha)*(level+trend)
            trend = self.beta*(level-last_level) + (1-self.beta)*trend
            result.append(level+trend)

        return result

    def __initial_trend__(self, series, slen):
        sum = 0.0
        for i in range(slen):
            sum += float(series[i+slen] - series[i]) / slen
        return sum / slen

    def additive(self, ts, forecast = False):

        def initial_seasonal_components(series, slen):
            seasonals = {}
            season_averages = []
            n_seasons = int(len(series)/slen)

            for j in range(n_seasons):
                season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))

            for i in range(slen):
                sum_of_vals_over_avg = 0.0
                for j in range(n_seasons):
                    sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
                seasonals[i] = sum_of_vals_over_avg/n_seasons

            return seasonals

        slen = get_frequency(ts)
        result = []
        seasonals = initial_seasonal_components(ts, slen)

        if forecast == True:
            x = 1
        else:
            x = 0

        for i in range(len(ts) + x):
            if i == 0:
                smooth = ts[0]
                trend = self.__initial_trend__(ts, slen)
                result.append(ts[0])
                continue
            if i >= len(ts):
                m = i - len(ts) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = ts[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])

        return result

    def multiplicative(self, ts, forecast = False):

        def initial_seasonal_components(series, slen):
            seasonals = {}
            season_averages = []
            n_seasons = int(len(series)/slen)

            for j in range(n_seasons):
                season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))

            for i in range(slen):
                sum_of_vals_over_avg = 0.0
                for j in range(n_seasons):
                    sum_of_vals_over_avg += series[slen*j+i]/season_averages[j]
                seasonals[i] = sum_of_vals_over_avg/n_seasons

            return seasonals

        slen = get_frequency(ts)
        result = []
        seasonals = initial_seasonal_components(ts, slen)

        if forecast == True:
            x = 1
        else:
            x = 0

        for i in range(len(ts) + x):
            if i == 0:
                smooth = ts[0]
                trend = self.__initial_trend__(ts, slen)
                result.append(ts[0])
                continue
            if i >= len(ts):
                m = i - len(ts) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = ts[i]
                last_smooth, smooth = smooth, self.alpha*(val/seasonals[i%slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%slen] = self.gamma*(val/smooth) + (1-self.gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])

        return result


    def predict(self, ts):
        if self.alpha != None and self.beta == False and self.gamma == False:
            result = self._alpha(ts)

        elif self.alpha == None and self.beta == False and self.gamma == False:
            self.fit(ts)
            result = self._alpha(ts)

        elif self.alpha != None and self.beta != None and (self.gamma == False or self.gamma == None):
            result = self._alpha_beta(ts)

        elif self.alpha == None and self.beta == None and (self.gamma == False or self.gamma == None):
            self.fit(ts)
            result = self._alpha_beta(ts)

        elif self.alpha != None and self.beta != None and self.gamma != None:
            if self.seasonal == 'multiplicative':
                result = self.multiplicative(ts)
            else:
                result = self.additive(ts)

        prediction = pandas.Series((v for v in result), index = ts.index)
        return prediction

    def fit(self, ts, error_function = None):

        if self.fit_model == 'no fit':
            pass
        else:
            def f(x):
                self.vector2params(x)
                return self.calc_error(ts, error_function)

            if self.alpha == None:
                self.alpha = numpy.random.rand(1)[0]
            if self.beta == None:
                self.beta = numpy.random.rand(1)[0]
            if self.gamma == None:
                self.gamma = numpy.random.rand(1)[0]

            x0 = self.params2vector()
            optim_params = scipy.optimize.minimize(f, x0)
            self.vector2params(vector = optim_params.x)

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

                if self.alpha != None and self.beta == False and self.gamma == False:
                    result = self._alpha(y, forecast = True)

                elif self.alpha != None and self.beta != None and self.gamma == False:
                    result = self._alpha_beta(y, forecast = True)

                elif self.alpha != None and self.beta != None and self.gamma != None:
                    if self.seasonal == 'multiplicative':
                        result = self.multiplicative(y, forecast = True)
                    else:
                        result = self.additive(y, forecast = True)
                next_value = result[-1]
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

        for i in range(periods):
            if i == 0:
                y = ts

            if self.alpha != None and self.beta == False and self.gamma == False:
                result = self._alpha(y, forecast = True)

            elif self.alpha != None and self.beta != None and self.gamma == False:
                result = self._alpha_beta(y, forecast = True)

            elif self.alpha != None and self.beta != None and self.gamma != None:
                if self.seasonal == 'multiplicative':
                    result = self.multiplicative(y, forecast = True)
                else:
                    result = self.additive(y, forecast = True)

            value = result[-1]
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
