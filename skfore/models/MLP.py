"""

Multi Layer Perceptron Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains a Multi Layer Perceptron model based on SciKit's MLPRegressor
model.


Examples
-------------------------------------------------------------------------------

Get predicted values as a DataFrame:

Load time series
>>> ts = load_champagne()

>>> model = MLP(p = 3, optim_type='auto')
>>> model
MLP(p = 3, model = None)

>>> random.seed(100)
>>> model.fit(ts) # doctest: +ELLIPSIS
MLP(p = 3, model = MLPRegressor(activation='relu', alpha=0.0005901, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(11,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=5000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False))

>>> random.seed(100)
>>> model.predict(ts, periods = 2) # doctest: +ELLIPSIS
                 ci_inf       ci_sup  ...      forecast  real
1972-10-01  3438...  7854...  ...   5609...  None
1972-11-01  4007...  7835...  ...   6230... None
<BLANKLINE>
[2 rows x 6 columns]





Classes
-------------------------------------------------------------------------------

"""

from skfore.base_model import base_model

import sklearn
from sklearn import neural_network

class MLP(base_model):
    """ Multi Layer Perceptron Model

    Args:
        p (int): order
        optim_type (str): 'auto' for alpha and hidden_layer_sizes selection or
            'use_parameters' to use set parameters
        **kwargs : sklearn.neural_network.MLPRegressor parameters

    Returns:
        MLP model structure of p lags and MLPRegressor model parameters

    """

    def __init__(self, p=None, optim_type='use_parameters', hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10):

        if p == None:
            self.p = 0
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

        self.optim_type = optim_type


    def __repr__(self):
        return 'MLP(p = ' + str(self.p) + ', model = ' + str(self.model) +')'


    def __get_X__(self, ts):
        try:
            y = ts.values
        except:
            y = ts
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

    def forecast(self, y):
        """ Next step

        Args:
            y (list): Time series list to find next value

        Returns:
            Value of next time stamp

        """
        Xtest = self.__get_X__(y)
        result = self.model.predict(Xtest)
        return result[-1]


    def simulate(self, ts):
        """ Fits a time series using self model parameters

        Args:
            ts (pandas.Series): Time series to fit

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
            ts (pandas.Series): Time series to fit
            error_function (function): Function to estimates error

        Return:
            self

        """

        if self.optim_type == 'auto':
             X = self.__get_X__(ts)
             y = ts
             opterror = None
             optmodel = None
             for alpha in numpy.arange(0.0000001,0.001,0.00001):
                 for neu in numpy.arange(1,self.p+10,1):
                     model = neural_network.MLPRegressor(hidden_layer_sizes=(neu, ), activation=self.activation, solver=self.solver, alpha=alpha, batch_size=self.batch_size, learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init, power_t=self.power_t, max_iter=5000, shuffle=self.shuffle, random_state=self.random_state, tol=self.tol, verbose=self.verbose, warm_start=self.warm_start, momentum=self.momentum, nesterovs_momentum=self.nesterovs_momentum, early_stopping=self.early_stopping, validation_fraction=self.validation_fraction, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
                     model.fit(X, y)
                     prediction = model.predict(X)
                     predicted = pandas.Series((v for v in prediction), index = y.index)
                     if (error_function == None):
                         error = sklearn.metrics.mean_squared_error(y[self.p:], predicted[self.p:])
                     else:
                         error = error_function(y[self.p:], predicted[self.p:])
                     if opterror is None or opterror > error:
                         opterror = error
                         optmodel = model
                         print('alpha = ' + str(alpha), 'hidden_layer_sizes = (' + str(neu) + ', )', 'error = ' + str(opterror))
             self.model = optmodel
        elif self.optim_type == 'use_parameters':
             X = self.__get_X__(ts)
             y = ts
             mlp_model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, batch_size=self.batch_size, learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init, power_t=self.power_t, max_iter=self.max_iter, shuffle=self.shuffle, random_state=self.random_state, tol=self.tol, verbose=self.verbose, warm_start=self.warm_start, momentum=self.momentum, nesterovs_momentum=self.nesterovs_momentum, early_stopping=self.early_stopping, validation_fraction=self.validation_fraction, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
             mlp_model.fit(X, y)
             self.model = mlp_model
        elif self.optim_type == 'no_optim':
            pass
        else:
            error_message = "Can't apply MLPRegressor using given parameters"
            raise ValueError(error_message)

        return self



if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.ELLIPSIS)
