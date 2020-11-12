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

Get predicted values as a DataFrame:
   
Load time series
>>> ts = load_champagne()

>>> model = AR(p = 3)
>>> model
AR(p = 3, intercept = None, phi = None)

>>> model.fit(ts) # doctest: +ELLIPSIS
AR(p = 3, intercept = ..., phi = [-0.0... -0.1...  0.5...])

>>> fitted_model = model.simulate(ts)
>>> model.predict(ts, periods = 2) # doctest: +ELLIPSIS
            ci_inf  ci_sup       series    bootstrap
1972-10-01     NaN     NaN  6...  6...
1972-11-01     NaN     NaN  5...  5...

If confidence intervals are calculated with 95% level and 300 iterations:
>>> model.predict(ts, periods = 2, confidence_interval = 0.95) # doctest: +ELLIPSIS
                 ci_inf        ci_sup    bootstrap       series
1972-10-01  ...  ...  6...  6...
1972-11-01  ...  ...  5...  5...

AR model using SciKit's Ridge linear model:
    
>>> model = AR_Ridge(p = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.simulate(ts)
>>> prediction = model.predict(ts, periods = 2)
>>> prediction # doctest: +ELLIPSIS
            ci_inf  ci_sup       series    bootstrap
1972-10-01     NaN     NaN  6...  5...
1972-11-01     NaN     NaN  5...  5...

AR model using SciKit's Lasso linear model:
    
>>> model = AR_Lasso(p = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.simulate(ts)
>>> prediction = model.predict(ts, periods = 2)
>>> prediction # doctest: +ELLIPSIS
            ci_inf  ci_sup       series    bootstrap
1972-10-01     NaN     NaN  6...  6...
1972-11-01     NaN     NaN  5...  5...

AR model using SciKit's Elastic Net linear model:
    
>>> model = AR_ElasticNet(p = 3)
>>> model = model.fit(ts)
>>> fitted_model = model.simulate(ts)
>>> prediction = model.predict(ts, periods = 2)
>>> prediction # doctest: +ELLIPSIS
            ci_inf  ci_sup       series    bootstrap
1972-10-01     NaN     NaN  6...  6...
1972-11-01     NaN     NaN  5...  5...

AR SciKit's models receives same parameters as regression models. 


Classes
-------------------------------------------------------------------------------

"""

from base_model import base_model
from datasets import *

import numpy
import scipy
import pandas
from extras import add_next_date
from sklearn import linear_model

class AR(base_model):
    """ Autoregressive model
    
    Parameter optimization method: scipy's minimization

    Args:
        p (int): order
        intercept (boolean or double): False for set intercept to 0 or double 
        phi (array): array of p-length for set parameters without optimization

    Returns:
        AR model structure of order p

    """

    def __init__(self, p=None, intercept=None, phi=None):
        
        
        if p == None:
            self.p = 1
        else:
            self.p = p
        
        
        if intercept == False:
            self.phi0 = 0
        else:
            self.phi0 = intercept
            
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
            None
            
        Returns:
            Vector parameters of length p+1 to use in optimization

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
            the model
            
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

    def get_X(self, ts):
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
    
    def forecast(self, ts):
        """ Next step 
        
        Args:
            ts (pandas.Series): Time series to find next value
            
        Returns:
            Value of next time stamp
            
        """
        
        if len(self.phi) != self.p:
            self.phi = numpy.random.rand(self.p)        
        
        y = ts.values
        lon = len(y)
        if lon <= self.p:
            y_last = y[0:lon]
            result = self.phi0 + numpy.dot(y_last, self.phi[0:lon])
        else:
            y_last = y[lon-self.p:lon]
            result = self.phi0 + numpy.dot(y_last, self.phi)

        return result

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
            if i == 0:
                result = self.phi0
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
            def f(x):
                self.vector2params(x)
                return self.calc_error(ts, error_function)

            x0 = self.params2vector()
            optim_params = scipy.optimize.minimize(f, x0)
            self.vector2params(vector = optim_params.x)

        return self


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

if __name__ == "__main__":
    import doctest
    doctest.testmod()