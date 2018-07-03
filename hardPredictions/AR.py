"""

AR Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains AR models using four diferent parameter optimization 
methods: SciPy's minimization, SciKit's Ridge linear model, SciKit's Lasso 
linear model and SciKit's Elastic Net linear model.


Classes
-------------------------------------------------------------------------------

"""

from hardPredictions.base_model import base_model

import numpy
import scipy
import pandas
import hardPredictions.extras
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit

class AR(base_model):
    """ Parameter optimization method: scipy's minimization 
    
    Args:
        p (int): order.
        
    Returns:
        AR model structure of order p.
        
    **Example**
    
    >>> ts = pandas.Series.from_csv('champagne.csv', index_col = 0, header = 0)
    >>> model = AR(p = 3)
    >>> model = model.fit(ts)
    >>> fitted_model = model.predict(ts)
    >>> prediction = model.forecast(ts, periods = 2)
    >>> prediction
    1972-10-01    6100.380339
    1972-11-01    5637.974302
    dtype: float64
    
    """
    
    def __init__(self, p=None):
        self.p = p
        self.phi0 = numpy.random.rand(1)
        self.phi = numpy.random.rand(p)
        
    def params2vector(self):
        params = list()
        params.append(self.phi0)
        for i in range(len(self.phi)):
            params.append(self.phi[i])
        return params
    
    def vector2params(self, vector):
        self.phi0 = vector[0]
        self.phi = vector[1:]
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
    
    def predict(self, ts):
        y = ts.values
        prediction = list()
        for i in range(len(y)):
            if i <= self.p:
                if i == 0:
                    prediction.append(self.phi0)
                else:
                    y_last = y[0:i]
                    result = self.phi0 + numpy.dot(y_last, self.phi[0:i])
                    prediction.append(result)
            else:
                y_last = y[i-self.p:i]
                result = self.phi0 + numpy.dot(y_last, self.phi)
                prediction.append(result)
        prediction = pandas.Series((v for v in prediction), index = ts.index)
        return prediction    
    
    
    def fit(self, ts, error_type = 'mean_squared_error'):
    
        def f(x):
            self.vector2params(x)            
            return self.calc_error(ts, error_type)
        
        x0 = self.params2vector()
        optim_params = scipy.optimize.minimize(f, x0)
        self.vector2params(vector = optim_params.x)   
        self.ts = self.predict(ts)  
        
        return self
    
    def forecast(self, ts, periods):
        
        def forward(y):
            y = y.values
            lon = len(y)
            if lon <= self.p:
                y_last = y[0:lon]
                result = self.phi0 + numpy.dot(y_last, self.phi[0:lon])
            else:
                y_last = y[lon-self.p:lon]
                result = self.phi0 + numpy.dot(y_last, self.phi)
            
            return result
        
        for i in range(periods):  
            if i == 0:
                y = ts
                value = forward(y)
            
            value = forward(y)
            y = hardPredictions.extras.add_next_date(y, value)
        
        return y[-periods:]
    
    def cross_validation(self, ts, n_splits, error_type = None):
        X = numpy.array(self.get_X(ts))
        y = numpy.array(ts.values.tolist())
        y_index = numpy.array(ts.index)
        tscv = TimeSeriesSplit(n_splits = n_splits)
        splits = tscv.split(X)
        
        error_list = list()
        for train_index, test_index in splits:
            #X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_index, y_test_index = y_index[train_index], y_index[test_index]
            
            y_train = pandas.Series((v for v in y_train), index = y_train_index)
            y_test = pandas.Series((v for v in y_test), index = y_test_index)
            self.fit(y_train)
            error = self.calc_error(y_test, error_type)
            error_list.append(error)
            
        return error_list

    def simulate(self, ts):
        pass
            
            
    
    
class AR_Ridge(AR):
    """ Parameter optimization method: SciKit's Ridge linear model """
    
    def __init__(self, p=None, alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, 
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
    
    def fit(self, ts):
        
        X = self.get_X(ts)
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
        self.ts = self.predict(ts)        
        
        return self
    
class AR_Ridge_2(AR):
    """ Parameter optimization method: SciKit's Ridge linear model """
    
    def __init__(self, p=None, **kwargs):
        self.p = p
    
    def fit(self, ts, **kwargs):
        
        X = self.get_X(ts)
        y = ts.values.tolist()
        ridge_model = linear_model.Ridge(**kwargs)
        ridge_model.fit(X, y)
        optim_params = list()
        optim_params.append(ridge_model.intercept_)
        optim_params = optim_params + ridge_model.coef_.tolist()
        self.vector2params(vector = optim_params)
        self.ts = self.predict(ts)        
        
        return self
    

class AR_Lasso(AR):
    """ Parameter optimization method: SciKit's Lasso linear model """
    
    def __init__(self, p=None, alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
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
    
    def fit(self, ts):
        
        X = self.get_X(ts)
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
        self.ts = self.predict(ts)        
        
        return self
    


class AR_ElasticNet(AR):
    """ Parameter optimization method: SciKit's Elastic Net linear model """
    
    def __init__(self, p=None, alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
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
    
    def fit(self, ts):
        
        X = self.get_X(ts)
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
        self.ts = self.predict(ts)        
        
        return self