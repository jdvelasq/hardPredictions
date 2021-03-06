
from skfore.models.BaseModel import BaseModel

import numpy
import scipy
import pandas
import sklearn

class Reg2TS(BaseModel):
    """ Regression to time series model


    """

    def __init__(self, p=None, model=None, **kwargs):


        if p == None:
            self.p = 0
        else:
            self.p = p


        if model == None:
            self.model = sklearn.linear_model.LinearRegression()
        else:
            self.model = model

        self.kwargs = kwargs


    def __repr__(self):
        return 'Reg2TS(p = ' + str(self.p) + ', model = ' + str(self.model) +')'



    def __get_X__(self, ts):
        """ Get matrix of regressors

        Args:
            ts (pandas.Series): Time series to create matrix of regressors

        Returns:
            List of list of regressors for every time in series

        """
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



    def fit(self, ts):

        X = self.__get_X__(ts)
        y = ts.values.tolist()
        fit_model = self.model(**self.kwargs)
        fit_model.fit(X, y)
        self.model = fit_model

        return self

