"""

MA Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains MA models using four diferent parameter optimization
methods: SciPy's minimization, SciKit's Ridge linear model, SciKit's Lasso
linear model and SciKit's Elastic Net linear model.

Examples
-------------------------------------------------------------------------------

MA model using SciPy's minimization:

>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = MA(q = 3)
>>> model
MA(q = 3, intercept = None, theta = None)
>>> model = model.fit(ts)
>>> model
MA(q = 3, intercept = 0.7576070305877793, theta = [0.47415837 0.96800789 0.50682355])
>>> prediction = model.forecast(ts, periods = 3)
>>> prediction
            ci_inf  ci_sup       series
1972-10-01     NaN     NaN  3644.781559
1972-11-01     NaN     NaN  4762.075950
1972-12-01     NaN     NaN  4204.870830
>>> prediction = model.forecast(ts, periods = 3, confidence_interval = 0.95)
>>> prediction
                 ci_inf       ci_sup       series
1972-10-01  -254.905599  6322.683321  3645.819521
1972-11-01   464.706361  7951.797533  4762.630139
1972-12-01 -1114.609601  8133.907219  4205.907253
>>> model.plot(ts, periods = 3, confidence_interval = 0.95)

.. image:: ./images/ma_1.png
  :width: 400
  :alt: MA 3
  :align: center
  



"""



import numpy
import scipy
import pandas
from sklearn import linear_model

from hardPredictions.base_model import base_model
from hardPredictions.extras import add_next_date


class MA(base_model):
    """ Moving-average model
    
    Parameter optimization method: scipy's minimization

    Args:
        q (int): order.

    Returns:
        MA model structure of order q.

    """

    def __init__(self, q=None, intercept=None, theta=None):
        self.y = None
        self.q = q
        
        if intercept == None:
            self.theta0 = None
        elif intercept == False:
            self.theta0 = 0
        else:
            self.theta0 = intercept
            
        if theta == None:
            self.theta = None
        else:
            self.theta = theta
            
        if intercept == None and theta == None:
            self.optim_type = 'complete'
        elif intercept == None and theta != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and theta == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and theta == None:
            self.optim_type = 'optim_params'
        elif intercept != None and theta != None:
            self.optim_type = 'no_optim'
            
        
    def __repr__(self):
        return 'MA(q = ' + str(self.q) + ', intercept = ' + str(self.theta0) + ', theta = ' + str(self.theta) +')'
        

    def params2vector(self):
        """ Parameters to vector
        
        Args:
            None.
            
        Returns:
            Vector parameters of length q+1 to use in optimization.

        """        
        params = list()
        if self.theta0 == None:
            self.theta0 = numpy.random.rand(1)[0]
        if self.theta == None:
            self.theta = numpy.random.rand(self.q)
        
        if self.optim_type == 'complete':
            params.append(self.theta0)
            for i in range(len(self.theta)):
                params.append(self.theta[i])
            return params
        elif self.optim_type == 'no_intercept' or self.optim_type == 'optim_params':
            for i in range(len(self.theta)):
                params.append(self.theta[i])
            return params
        elif self.optim_type == 'optim_intercept':
            params.append(self.theta0)
            return params
        elif self.optim_type == 'no_optim':
            pass
        

    def vector2params(self, vector):
        """ Vector to parameters
        
        Args:
            vector (list): vector of length q+1 to convert into parameters of 
            the model.
            
        Returns:
            self

        """ 
        
        if self.optim_type == 'complete':
            self.theta0 = vector[0]
            self.theta = vector[1:]
        elif self.optim_type == 'no_intercept' or self.optim_type == 'optim_params':
            self.theta = vector
        elif self.optim_type == 'optim_intercept':
            self.theta0 = vector[0]
        elif self.optim_type == 'no_optim':
            pass
            
        return self

    def __get_X__(self, ts):
        if self.y == None:
            lon = len(ts.values)
            y = numpy.random.randn(lon)
        else:
            y = self.y
        X = list()
        for i in range(len(ts)):
            if i <= self.q:
                if i == 0:
                    value = [0] * self.q
                    X.append(value)
                else:
                    value_0 = [0] * (self.q - i)
                    value_1 = y[0:i].tolist()
                    value = value_0 + value_1
                    X.append(value)
            else:
                value = y[i-self.q:i].tolist()
                X.append(value)
        return X
    
    def __forward__(self, ts):
        if self.y == None:
            lon = len(ts.values)
            y = numpy.random.randn(lon)
        else:
            y = self.y
        lon = len(y)
        if lon <= self.q:
             y_last = y[0:lon]
             ts_last = ts[0:lon+1]
             mean_ts = ts_last.mean()                    
             result = self.theta0 + mean_ts + numpy.dot(y_last, self.theta[0:lon])
        else:
            y_last = y[lon-self.q:lon]
            ts_last = ts[lon-self.q+1:lon+1]
            mean_ts = ts_last.mean() 
            result = self.theta0 + mean_ts + numpy.dot(y_last, self.theta)

        return result

    def predict(self, ts):
        """ Fits a time series using self model parameters
        
        Args:
            ts (pandas.Series): Time series to fit.
        
        Returns:
            Fitted time series.
            
        """
        if self.y == None:
            lon = len(ts.values)
            y = numpy.random.randn(lon)
        else:
            y = self.y
        #y = ts
        prediction = list()
        for i in range(len(y)):
            if i == 0:
                result = self.theta0
            else:
                lon = len(prediction)
                if lon <= self.q:
                    y_last = y[0:lon]
                    ts_last = ts[0:lon+1]
                    mean_ts = ts_last.mean()                    
                    result = self.theta0 + mean_ts + numpy.dot(y_last, self.theta[0:lon])
                else:
                    y_last = y[lon-self.q:lon]
                    ts_last = ts[lon-self.q+1:lon+1]
                    mean_ts = ts_last.mean() 
                    result = self.theta0 + mean_ts + numpy.dot(y_last, self.theta)
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
    

class MA_Ridge(MA):
    """ Parameter optimization method: SciKit's Ridge linear model """

    def __init__(self, q=None, intercept=None, theta=None, alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
                 normalize=False, random_state=None, solver='auto', tol=0.001):
        self.y = None
        self.q = q
        self.alpha = alpha
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.normalize = normalize
        self.random_state = random_state
        self.solver = solver
        self.tol = tol
        
        if intercept == None:
            self.theta0 = numpy.random.rand(1)
        elif intercept == False:
            self.theta0 = 0
        else:
            self.theta0 = intercept
            
        if theta == None:
            self.theta = numpy.random.rand(q)
        else:
            self.theta = theta
        
        if intercept == None and theta == None:
            self.optim_type = 'complete'
        elif intercept == None and theta != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and theta == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and theta == None:
            self.optim_type = 'optim_params'
        elif intercept != None and theta != None:
            self.optim_type = 'no_optim'
    
    def __repr__(self):
        return 'MA_Ridge(q = ' + str(self.q) + ', intercept = ' + str(self.theta) + ', theta = ' + str(self.theta) +')'
        

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

class MA_Lasso(MA):
    """ Parameter optimization method: SciKit's Lasso linear model """

    def __init__(self, q=None, intercept=None, theta=None, alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                 normalize=False, positive=False, precompute=False, random_state=None,
                 selection='cyclic', tol=0.0001, warm_start=False):
        self.y = None
        self.q = q
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
            self.theta0 = numpy.random.rand(1)
        elif intercept == False:
            self.theta0 = 0
        else:
            self.theta0 = intercept
            
        if theta == None:
            self.theta = numpy.random.rand(q)
        else:
            self.theta = theta
        
        if intercept == None and theta == None:
            self.optim_type = 'complete'
        elif intercept == None and theta != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and theta == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and theta == None:
            self.optim_type = 'optim_params'
        elif intercept != None and theta != None:
            self.optim_type = 'no_optim'
    
    def __repr__(self):
        return 'MA_Lasso(q = ' + str(self.q) + ', intercept = ' + str(self.theta0) + ', theta = ' + str(self.theta) +')'
        

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



class MA_ElasticNet(MA):
    """ Parameter optimization method: SciKit's Elastic Net linear model """

    def __init__(self, q=None, intercept=None, theta=None, alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
                 max_iter=1000, normalize=False, positive=False, precompute=False,
                 random_state=0, selection='cyclic', tol=0.0001, warm_start=False):
        
        self.y = None
        self.q = q
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
            self.theta0 = numpy.random.rand(1)
        elif intercept == False:
            self.theta0 = 0
        else:
            self.theta0 = intercept
            
        if theta == None:
            self.theta = numpy.random.rand(q)
        else:
            self.theta = theta
        
        if intercept == None and theta == None:
            self.optim_type = 'complete'
        elif intercept == None and theta != None:
            self.optim_type = 'optim_intercept'
        elif intercept == False and theta == None:
            self.optim_type = 'no_intercept'
        elif intercept != None and theta == None:
            self.optim_type = 'optim_params'
        elif intercept != None and theta != None:
            self.optim_type = 'no_optim'
    
    def __repr__(self):
        return 'MA_ElasticNet(q = ' + str(self.q) + ', intercept = ' + str(self.theta0) + ', theta = ' + str(self.theta) +')'
        
    
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
