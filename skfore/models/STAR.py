"""

STAR Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains STAR models using SciPy's minimization method for parameter
optimization.

Examples
-------------------------------------------------------------------------------

Get predicted values as a DataFrame:

Load time series
>>> ts = load_champagne()

>>> model = STAR(p = 3)
>>> model
STAR(p = 3, gamma = None, center = None, intercept_1 = None, phi_1 = None, intercept_2 = None, phi_2 = None)

>>> random.seed(100)
>>> model.fit(ts) # doctest: +ELLIPSIS
STAR(p = 3, gamma = 0.627..., center = 0.567..., intercept_1 = 3567..., phi_1 = [-0.094...   -0.193...  0.549... ], intercept_2 = 0.970..., phi_2 = [0.231... 0.769... 0.990...])

>>> random.seed(100)
>>> model.predict(ts, periods = 2) # doctest: +ELLIPSIS
                 ci_inf        ci_sup  ...      forecast  real
1972-10-01  2497...  10038...  ...   6119...  None
1972-11-01  1974...  10220...  ...   5706...  None
<BLANKLINE>
[2 rows x 6 columns]



Classes
-------------------------------------------------------------------------------

"""

from skfore.base_model import base_model

class STAR(base_model):
    """ Smooth Transition Autoregressive model

    Parameter optimization method: scipy's minimization

    Args:
        p (int): order
        gamma (float): gamma parameter for transition function
        intercept_1 (boolean or double): Intercept for first model. False for set intercept to 0 or double
        intercept_2 (boolean or double): Intercept for second model. False for set intercept to 0 or double
        phi_1 (array): array of p-length for set parameters of first model without optimization
        phi_2 (array): array of p-length for set parameters of second model without optimization

    Returns:
        STAR model structure of order p

    """

    def __init__(self, p=None, gamma=None, center=None, intercept_1=None, intercept_2 =None, phi_1=None, phi_2=None):
        if p == None:
            raise ValueError('Please insert parameter p')
        else:
            self.p = p


        self.gamma = gamma
        self.center = center
        self.intercept_1 = intercept_1
        self.intercept_2 = intercept_2
        self.phi_1 = phi_1
        self.phi_2 = phi_2

        self.gamma_value = (gamma == None)
        self.center_value = (center == None)
        self.intercept_1_value = (intercept_1 == None)
        self.phi_1_value = (phi_1 == None)
        self.intercept_2_value = (intercept_2 == None)
        self.phi_2_value = (phi_2 == None)

        self.optim_type = 'complete'



    def __repr__(self):
        return 'STAR(p = ' + str(self.p) + ', gamma = ' + str(self.gamma) + ', center = ' + str(self.center) +  ', intercept_1 = ' + str(self.intercept_1) +  ', phi_1 = ' + str(self.phi_1) +  ', intercept_2 = ' + str(self.intercept_2) + ', phi_2 = ' + str(self.phi_2) +')'


    def params2vector(self):
        """ Parameters to vector

        Args:
            None

        Returns:
            Vector parameters of length p+1 to use in optimization

        """
        params = list()
        if self.gamma == None:
            self.gamma = numpy.random.rand(1)[0]
        if self.center == None:
            self.center = numpy.random.rand(1)[0]
        if self.intercept_1 == None:
            self.intercept_1 = numpy.random.rand(1)[0]
        if self.phi_1 == None:
            self.phi_1 = numpy.random.rand(self.p)
        if self.intercept_2 == None:
            self.intercept_2 = numpy.random.rand(1)[0]
        if self.phi_2 == None:
            self.phi_2 = numpy.random.rand(self.p)

        if self.optim_type == 'complete':
            if self.gamma_value:
                params.append(self.gamma)
            if self.center_value:
                params.append(self.center)
            if self.intercept_1_value:
                params.append(self.intercept_1)
            if self.phi_1_value:
                for i in range(len(self.phi_1)):
                    params.append(self.phi_1[i])
            if self.intercept_2_value:
                params.append(self.intercept_2)
            if self.phi_2_value:
                for i in range(len(self.phi_2)):
                    params.append(self.phi_2[i])
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
            i = 0
            if self.gamma_value:
                self.gamma = vector[i]
                i = i+1
            if self.center_value:
                self.center = vector[i]
                i = i+1
            if self.intercept_1_value:
                self.intercept_1 = vector[i]
                i = i+1
            if self.phi_1_value:
                self.phi_1 = vector[i:i+self.p]
                i = i+1
            if self.intercept_2_value:
                self.intercept_2 = vector[3+self.p]
                i = i+1
            if self.phi_2_value:
                self.phi_2 = vector[4+self.p:4+2*self.p]
        elif self.optim_type == 'no_optim':
            pass

        return self

    def forecast(self, y):
        """ Next step

        Args:
            ts (pandas.Series): Time series to find next value

        Returns:
            Value of next time stamp

        """
        y = y.values
        lon = len(y)
        if lon <= self.p:
            z_t = numpy.mean(y)
            G_zt = 1/(1+math.exp(-1*self.gamma*(z_t-self.center)))
            y_last = y[0:lon]
            result = self.intercept_1*G_zt + numpy.dot(y_last, self.phi_1[0:lon])*G_zt + self.intercept_2*(1-G_zt) + numpy.dot(y_last, self.phi_2[0:lon])*(1-G_zt)
        else:
            z_t = y[-(self.p+1)]
            G_zt = 1/(1+math.exp(-1*self.gamma*(z_t-self.center)))
            y_last = y[lon-self.p:lon]
            result = self.intercept_1*G_zt + numpy.dot(y_last, self.phi_1)*G_zt + self.intercept_2*(1-G_zt) + numpy.dot(y_last, self.phi_2)*(1-G_zt)

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
            result = self.forecast(y[0:i])
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
                return self.calc_error(ts, error_function, self.p)

            x0 = self.params2vector()
            optim_params = scipy.optimize.minimize(f, x0)
            self.vector2params(vector = optim_params.x)

        return self

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.ELLIPSIS)
