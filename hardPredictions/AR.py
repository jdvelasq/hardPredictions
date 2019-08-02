"""

AR Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains AR models using four diferent parameter optimization
methods: SciPy's minimization, SciKit's Ridge linear model, SciKit's Lasso
linear model and SciKit's Elastic Net linear model.

Examples
-------------------------------------------------------------------------------

AR model using SciPy's minimization:

>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = AR(p = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.predict(ts)
>>> prediction = model.forecast(ts, periods = 2)
>>> prediction
1972-10-01    6100.380339
1972-11-01    5637.974302
dtype: float64

AR model using SciKit's Ridge linear model:
    
>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = AR_Ridge(p = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.predict(ts)
>>> prediction = model.forecast(ts, periods = 2)
>>> prediction
1972-10-01    6056.234637
1972-11-01    5514.641861
dtype: float64

AR model using SciKit's Lasso linear model:
    
>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = AR_Lasso(p = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.predict(ts)
>>> prediction = model.forecast(ts, periods = 2)
>>> prediction
1972-10-01    6056.234513
1972-11-01    5514.641777
dtype: float64

AR model using SciKit's Elastic Net linear model:
    
>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = AR_ElasticNet(p = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.predict(ts)
>>> prediction = model.forecast(ts, periods = 2)
>>> prediction
1972-10-01    6056.233741
1972-11-01    5514.641325
dtype: float64       


Classes
-------------------------------------------------------------------------------

"""

from base_model import base_model

import numpy
import scipy
import pandas
import matplotlib
from extras import add_next_date
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample

class AR(base_model):
    """ Autoregressive model
    
    Parameter optimization method: scipy's minimization

    Args:
        p (int): order.

    Returns:
        AR model structure of order p.

    """

    def __init__(self, p=None, intercept=None, phi=None):
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p
        
        if intercept == None:
            self.phi0 = None
        elif intercept == False:
            self.phi0 = 0
        else:
            self.phi0 = intercept
            
        if phi == None:
            self.phi = None
        else:
            self.phi = phi
            
        if intercept == None and phi == None:
            self.optim_type = 'complete'
        elif intercept == None and phi != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and phi == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and phi == None:
            self.optim_type = 'optim_params'
        elif intercept != None and phi != None:
            self.optim_type = 'no_optim'
            
        
    def __repr__(self):
        return 'AR(p = ' + str(self.p) + ', intercept = ' + str(self.phi0) + ', phi = ' + str(self.phi) +')'
        

    def params2vector(self):
        """ Parameters to vector
        
        Args:
            None.
            
        Returns:
            Vector parameters of length p+1 to use in optimization.

        """        
        params = list()
        if self.phi0 == None:
            self.phi0 = numpy.random.rand(1)[0]
        if self.phi == None:
            self.phi = numpy.random.rand(self.p)
        
        if self.optim_type == 'complete':
            params.append(self.phi0)
            for i in range(len(self.phi)):
                params.append(self.phi[i])
            return params
        elif self.optim_type == 'no_intercept' or self.optim_type == 'optim_params':
            for i in range(len(self.phi)):
                params.append(self.phi[i])
            return params
        elif self.optim_type == 'optim_intercept':
            params.append(self.phi0)
            return params
        elif self.optim_type == 'no_optim':
            pass
        

    def vector2params(self, vector):
        """ Vector to parameters
        
        Args:
            vector (list): vector of length p+1 to convert into parameters of 
            the model.
            
        Returns:
            self

        """ 
        
        if self.optim_type == 'complete':
            self.phi0 = vector[0]
            self.phi = vector[1:]
        elif self.optim_type == 'no_intercept' or self.optim_type == 'optim_params':
            self.phi = vector
        elif self.optim_type == 'optim_intercept':
            self.phi0 = vector[0]
        elif self.optim_type == 'no_optim':
            pass
            
        return self

    def __get_X__(self, ts):
        y = ts.values
        X = list()
        for i in range(len(ts)):
            if i <= self.p:
                if i == 0:
                    value = [0] * self.p
                    X.append(value)
                else:
                    value_0 = [0] * (self.p - i)
                    value_1 = y[0:i].tolist()
                    value = value_0 + value_1
                    X.append(value)
            else:
                value = y[i-self.p:i].tolist()
                X.append(value)
        return X
    
    def __forward__(self, y):
        y = y.values
        lon = len(y)
        if lon <= self.p:
            y_last = y[0:lon]
            result = self.phi0 + numpy.dot(y_last, self.phi[0:lon])
            #result = numpy.nan
        else:
            y_last = y[lon-self.p:lon]
            result = self.phi0 + numpy.dot(y_last, self.phi)

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
            if i == 0:
                result = self.phi0
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
            def f(x):
                self.vector2params(x)
                return self.calc_error(ts, error_function)

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



class AR_Ridge(AR):
    """ Parameter optimization method: SciKit's Ridge linear model """

    def __init__(self, p=None, intercept=None, phi=None, alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
                 normalize=False, random_state=None, solver='auto', tol=0.001):
        self.p = p
        self.alpha = alpha
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.normalize = normalize
        self.random_state = random_state
        self.solver = solver
        self.tol = tol
        
        if intercept == None:
            self.phi0 = numpy.random.rand(1)
        elif intercept == False:
            self.phi0 = 0
        else:
            self.phi0 = intercept
            
        if phi == None:
            self.phi = numpy.random.rand(p)
        else:
            self.phi = phi
        
        if intercept == None and phi == None:
            self.optim_type = 'complete'
        elif intercept == None and phi != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and phi == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and phi == None:
            self.optim_type = 'optim_params'
        elif intercept != None and phi != None:
            self.optim_type = 'no_optim'
    
    def __repr__(self):
        return 'AR_Ridge(p = ' + str(self.p) + ', intercept = ' + str(self.phi0) + ', phi = ' + str(self.phi) +')'
        

    def fit(self, ts):
        
        if self.optim_type == 'complete':
             X = self.__get_X__(ts)
             y = ts.values.tolist()
             ridge_model = linear_model.Ridge(alpha = self.alpha, copy_X = self.copy_X,
                                              fit_intercept = self.fit_intercept,
                                              max_iter = self.max_iter,
                                              normalize = self.normalize,
                                              random_state = self.random_state,
                                              solver = self.solver, tol = self.tol)
             ridge_model.fit(X, y)
             optim_params = list()
             optim_params.append(ridge_model.intercept_)
             optim_params = optim_params + ridge_model.coef_.tolist()
             self.vector2params(vector = optim_params)
        
        elif self.optim_type == 'no_intercept':
            X = self.__get_X__(ts)
            y = ts.values.tolist()
            ridge_model = linear_model.Ridge(alpha = self.alpha, copy_X = self.copy_X,
                                             fit_intercept = False,
                                             max_iter = self.max_iter,
                                             normalize = self.normalize,
                                             random_state = self.random_state,
                                             solver = self.solver, tol = self.tol)
            ridge_model.fit(X, y)
            optim_params = list()
            optim_params = optim_params + ridge_model.coef_.tolist()
            self.vector2params(vector = optim_params)
        
        elif self.optim_type == 'no_optim':
            pass
        else:
            error_message = "Can't apply Lasso regression using given parameters"
            raise ValueError(error_message) 
        
        return self   


#class AR_Ridge_2(AR):
#    """ Parameter optimization method: SciKit's Ridge linear model """

#   def __init__(self, p=None, **kwargs):
#        self.p = p

#    def fit(self, ts, **kwargs):

#        X = self.__get_X__(ts)
#        y = ts.values.tolist()
#        ridge_model = linear_model.Ridge(**kwargs)
#        ridge_model.fit(X, y)
#        optim_params = list()
#        optim_params.append(ridge_model.intercept_)
#        optim_params = optim_params + ridge_model.coef_.tolist()
#        self.vector2params(vector = optim_params)

#        return self


class AR_Lasso(AR):
    """ Parameter optimization method: SciKit's Lasso linear model """

    def __init__(self, p=None, intercept=None, phi=None, alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                 normalize=False, positive=False, precompute=False, random_state=None,
                 selection='cyclic', tol=0.0001, warm_start=False):
        self.p = p
        self.alpha = alpha
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.normalize = normalize
        self.positive = positive
        self.precompute = precompute
        self.random_state = random_state
        self.selection = selection
        self.tol = tol
        self.warm_start = warm_start
        
        if intercept == None:
            self.phi0 = numpy.random.rand(1)
        elif intercept == False:
            self.phi0 = 0
        else:
            self.phi0 = intercept
            
        if phi == None:
            self.phi = numpy.random.rand(p)
        else:
            self.phi = phi
        
        if intercept == None and phi == None:
            self.optim_type = 'complete'
        elif intercept == None and phi != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and phi == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and phi == None:
            self.optim_type = 'optim_params'
        elif intercept != None and phi != None:
            self.optim_type = 'no_optim'
    
    def __repr__(self):
        return 'AR_Lasso(p = ' + str(self.p) + ', intercept = ' + str(self.phi0) + ', phi = ' + str(self.phi) +')'
        

    def fit(self, ts):
        
        if self.optim_type == 'complete':
            X = self.__get_X__(ts)
            y = ts.values.tolist()
            lasso_model = linear_model.Lasso(alpha = self.alpha, copy_X = self.copy_X,
                                             fit_intercept = self.fit_intercept,
                                             max_iter = self.max_iter,
                                             normalize = self.normalize,
                                             positive = self.positive,
                                             precompute = self.precompute,
                                             random_state = self.random_state,
                                             selection = self.selection, tol = self.tol,
                                             warm_start = self.warm_start)
            lasso_model.fit(X, y)
            optim_params = list()
            optim_params.append(lasso_model.intercept_)
            optim_params = optim_params + lasso_model.coef_.tolist()
            self.vector2params(vector = optim_params)
        
        elif self.optim_type == 'no_intercept':
            X = self.__get_X__(ts)
            y = ts.values.tolist()
            lasso_model = linear_model.Lasso(alpha = self.alpha, copy_X = self.copy_X,
                                             fit_intercept = False,
                                             max_iter = self.max_iter,
                                             normalize = self.normalize,
                                             positive = self.positive,
                                             precompute = self.precompute,
                                             random_state = self.random_state,
                                             selection = self.selection, tol = self.tol,
                                             warm_start = self.warm_start)
            lasso_model.fit(X, y)
            optim_params = list()
            optim_params = optim_params + lasso_model.coef_.tolist()
            self.vector2params(vector = optim_params)
        
        elif self.optim_type == 'no_optim':
            pass
        else:
            error_message = "Can't apply Lasso regression using given parameters"
            raise ValueError(error_message) 
        
        return self        



class AR_ElasticNet(AR):
    """ Parameter optimization method: SciKit's Elastic Net linear model """

    def __init__(self, p=None, intercept=None, phi=None, alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
                 max_iter=1000, normalize=False, positive=False, precompute=False,
                 random_state=0, selection='cyclic', tol=0.0001, warm_start=False):
        
        self.p = p
        self.alpha = alpha
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.normalize = normalize
        self.positive = positive
        self.precompute = precompute
        self.random_state = random_state
        self.selection = selection
        self.tol = tol
        self.warm_start = warm_start
        
        if intercept == None:
            self.phi0 = numpy.random.rand(1)
        elif intercept == False:
            self.phi0 = 0
        else:
            self.phi0 = intercept
            
        if phi == None:
            self.phi = numpy.random.rand(p)
        else:
            self.phi = phi
        
        if intercept == None and phi == None:
            self.optim_type = 'complete'
        elif intercept == None and phi != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and phi == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and phi == None:
            self.optim_type = 'optim_params'
        elif intercept != None and phi != None:
            self.optim_type = 'no_optim'
    
    def __repr__(self):
        return 'AR_ElasticNet(p = ' + str(self.p) + ', intercept = ' + str(self.phi0) + ', phi = ' + str(self.phi) +')'
        
    
    def fit(self, ts):
        
        if self.optim_type == 'complete':
             X = self.__get_X__(ts)
             y = ts.values.tolist()
             lasso_model = linear_model.ElasticNet(alpha = self.alpha, copy_X = self.copy_X,
                                                   fit_intercept = self.fit_intercept,
                                                   l1_ratio = self.l1_ratio,
                                                   max_iter = self.max_iter,
                                                   normalize = self.normalize,
                                                   positive = self.positive,
                                                   precompute = self.precompute,
                                                   random_state = self.random_state,
                                                   selection = self.selection, tol = self.tol,
                                                   warm_start = self.warm_start)
             lasso_model.fit(X, y)
             optim_params = list()
             optim_params.append(lasso_model.intercept_)
             optim_params = optim_params + lasso_model.coef_.tolist()
             self.vector2params(vector = optim_params)
        
        elif self.optim_type == 'no_intercept':
             X = self.__get_X__(ts)
             y = ts.values.tolist()
             lasso_model = linear_model.ElasticNet(alpha = self.alpha, copy_X = self.copy_X,
                                                   fit_intercept = False,
                                                   l1_ratio = self.l1_ratio,
                                                   max_iter = self.max_iter,
                                                   normalize = self.normalize,
                                                   positive = self.positive,
                                                   precompute = self.precompute,
                                                   random_state = self.random_state,
                                                   selection = self.selection, tol = self.tol,
                                                   warm_start = self.warm_start)
             lasso_model.fit(X, y)
             optim_params = list()
             optim_params = optim_params + lasso_model.coef_.tolist()
             self.vector2params(vector = optim_params)
        
        elif self.optim_type == 'no_optim':
            pass
        else:
            error_message = "Can't apply Elastic Net regression using given parameters"
            raise ValueError(error_message) 
        
        return self  
