"""

Multi Layer Perceptron Model
===============================================================================

Overview
-------------------------------------------------------------------------------

Examples
-------------------------------------------------------------------------------


Classes
-------------------------------------------------------------------------------

"""

from base_model import base_model

import numpy
import scipy
import pandas
from extras import add_next_date
from sklearn import neural_network

class MLP(base_model):
    """ Multi Layer Perceptron Model
    
    Parameter optimization method: scipy's minimization

    Args:
        p (int): order.

    Returns:
        MLP model structure of order p.

    """

    def __init__(self, p=None, hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10):
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle 
        self.random_state = random_state
        self.tol = tol 
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum 
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping 
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        
        self.model = None
        
        self.optim_type = 'complete'
        
        
    def __repr__(self):
        return 'MLP(p = ' + str(self.p) + ', model = ' + str(self.model) +')'
        

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
        Xtest = self.__get_X__(y)
        result = self.model.predict(Xtest)
        return result[-1]
    

    def predict(self, ts):
        """ Fits a time series using self model parameters
        
        Args:
            ts (pandas.Series): Time series to fit.
        
        Returns:
            Fitted time series.
            
        """

        Xtest = self.__get_X__(ts)
        prediction = self.model.predict(Xtest)
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
        
        if self.optim_type == 'complete':
             X = self.__get_X__(ts)
             y = ts.values.tolist()
             mlp_model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, batch_size=self.batch_size, learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init, power_t=self.power_t, max_iter=self.max_iter, shuffle=self.shuffle, random_state=self.random_state, tol=self.tol, verbose=self.verbose, warm_start=self.warm_start, momentum=self.momentum, nesterovs_momentum=self.nesterovs_momentum, early_stopping=self.early_stopping, validation_fraction=self.validation_fraction, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
             mlp_model.fit(X, y)
             self.model = mlp_model      
        elif self.optim_type == 'no_optim':
            pass
        else:
            error_message = "Can't apply Lasso regression using given parameters"
            raise ValueError(error_message)
            
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