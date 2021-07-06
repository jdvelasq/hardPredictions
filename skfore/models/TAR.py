"""

TAR Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains TAR models using SciPy's minimization method for AR based
models.

Examples
-------------------------------------------------------------------------------

Get predicted values as a DataFrame:

Load time series
>>> ts = load_champagne()

>>> model = TAR(p = 3)
>>> model
TAR(p = 3, intercept_1 = None, intercept_2 = None, phi_1 = None, phi_2 = None)

>>> random.seed(1)
>>> model.fit(ts) # doctest: +ELLIPSIS
TAR(p = 3, intercept_1 = 2758..., intercept_2 = 4144..., phi_1 = [-0.106... -0.250...   0.776... ], phi_2 = [-0.074... -0.188...  0.479...])

>>> random.seed(1)
>>> model.predict(ts, periods = 2) # doctest: +ELLIPSIS
                 ci_inf       ci_sup  ...      forecast  real
1972-10-01  2435...  9979...  ...   6239...  None
1972-11-01  2150...  9655...  ...   5917...  None
<BLANKLINE>
[2 rows x 6 columns]



"""

from skfore.models.BaseModel import BaseModel

from skfore.models.AR import AR

class TAR(BaseModel):
    """ Threshold autoregressive for AR based models

    Parameter optimization method: scipy's minimization

    Args:
        p (int): order
        intercept_1 (boolean or double): Intercept for first model. False for set intercept to 0 or double
        intercept_2 (boolean or double): Intercept for second model. False for set intercept to 0 or double
        phi_1 (array): array of p-length for set parameters of first model without optimization
        phi_2 (array): array of p-length for set parameters of second model without optimization

    Returns:
        AR model structure of order p

    """

    def __init__(self, p=None, intercept_1=None, intercept_2 =None, phi_1=None, phi_2=None, max_interval = 10):


        self.p = p

        self.intercept_1 = intercept_1
        self.intercept_2 = intercept_2
        self.phi_1 = phi_1
        self.phi_2 = phi_2

        self.max_interval = max_interval
        self.d = max_interval
        self.ts_1 = None
        self.ts_2 = None



    def __repr__(self):
        return 'TAR(p = ' + str(self.p) + ', intercept_1 = ' + str(self.intercept_1) + ', intercept_2 = ' + str(self.intercept_2) + ', phi_1 = ' + str(self.phi_1) + ', phi_2 = ' + str(self.phi_2) +')'


    def simulate(self, ts):
        """ Fits a time series using self model parameters

        Args:
            ts (pandas.Series): Time series to fit

        Returns:
            Fitted time series

        """

        prediction_1 = self.model_1.simulate(self.ts_1)
        prediction_2 = self.model_2.simulate(self.ts_2)
        prediction = prediction_1.append(prediction_2)
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

        if self.p == None:
            raise ValueError('Please insert parameter p')

        errors = list()
        tes = list()
        for d in range(self.max_interval, len(ts)-self.max_interval):
            self.ts_1 = ts[0:d]
            self.ts_2 = ts[d:]
            self.model_1 = AR(p = self.p, intercept = self.intercept_1, phi = self.phi_1)
            self.model_1 = self.model_1.fit(self.ts_1, error_function)
            self.model_2 = AR(p = self.p, intercept = self.intercept_2, phi = self.phi_2)
            self.model_2 = self.model_2.fit(self.ts_2, error_function)
            error = self.calc_error(ts)
            errors.append(error)
            tes.append(d)
        minim = errors.index(min(errors))
        self.d = tes[minim]
        self.ts_1 = ts[0:self.d]
        self.ts_2 = ts[self.d:]
        self.model_1 = AR(p = self.p, intercept = self.intercept_1, phi = self.phi_1)
        self.model_1 = self.model_1.fit(self.ts_1, error_function)
        self.model_2 = AR(p = self.p, intercept = self.intercept_2, phi = self.phi_2)
        self.model_2 = self.model_2.fit(self.ts_2, error_function)

        self.intercept_1 = self.model_1.phi0
        self.intercept_2 = self.model_2.phi0
        self.phi_1 = self.model_1.phi
        self.phi_2 = self.model_2.phi

        return self


    def forecast(self, ts):
        """ Next step

        Args:
            ts (pandas.Series): Time series to find next value

        Returns:
            Value of next time stamp

        """

        return self.model_2.forecast(ts)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.ELLIPSIS)
