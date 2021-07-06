"""

Transformer: time series' transformation module
===============================================================================


Overview
-------------------------------------------------------------------------------
This module contains time series' transformation methods (log, log10, sqrt,
cbrt, boxcox), trending (linear, cuadratic, cubic, diff1, diff2) and seasonal
(poly2, diff) removal methods.

Examples
-------------------------------------------------------------------------------

Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trans = 'log')
>>> mytransform
Transformer(trans = log, trend = None, seasonal = None)
>>> transformed_ts = mytransform.fit(ts = ts)
>>> transformed_ts
Month
1964-01-01    7.942718
1964-02-01    7.890583
1964-03-01    7.921173
1964-04-01    7.908755
1964-05-01    7.988204
1964-06-01    8.018296
1964-07-01    7.732808
1964-08-01    7.701652
1964-09-01    7.980024
1964-10-01    8.366603
1964-11-01    8.659387
1964-12-01    8.897272
Name: Perrin, dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trans = 'log10').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01    3.449478
1964-02-01    3.426836
1964-03-01    3.440122
1964-04-01    3.434729
1964-05-01    3.469233
1964-06-01    3.482302
1964-07-01    3.358316
1964-08-01    3.344785
1964-09-01    3.465680
1964-10-01    3.633569
1964-11-01    3.760724
1964-12-01    3.864036
Name: Perrin, dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trans = 'sqrt').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01    53.056574
1964-02-01    51.691392
1964-03-01    52.488094
1964-04-01    52.163205
1964-05-01    54.277067
1964-06-01    55.099909
1964-07-01    47.770284
1964-08-01    47.031904
1964-09-01    54.055527
1964-10-01    65.582010
1964-11-01    75.921012
1964-12-01    85.510233
Name: Perrin, dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trans = 'cbrt').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01    14.119722
1964-02-01    13.876464
1964-03-01    14.018683
1964-04-01    13.960775
1964-05-01    14.335436
1964-06-01    14.479956
1964-07-01    13.165536
1964-08-01    13.029519
1964-09-01    14.296402
1964-10-01    16.262594
1964-11-01    17.929767
1964-12-01    19.409398
Name: Perrin, dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trans = 'boxcox').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01    0.504795
1964-02-01    0.504795
1964-03-01    0.504795
1964-04-01    0.504795
1964-05-01    0.504795
1964-06-01    0.504795
1964-07-01    0.504795
1964-08-01    0.504795
1964-09-01    0.504795
1964-10-01    0.504795
1964-11-01    0.504795
1964-12-01    0.504795
Name: Perrin, dtype: float64


Removing trend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trend = 'linear').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01     993.871795
1964-02-01     549.592075
1964-03-01     331.312354
1964-04-01      -3.967366
1964-05-01     -80.247086
1964-06-01    -291.526807
1964-07-01   -1346.806527
1964-08-01   -1718.086247
1964-09-01   -1309.365967
1964-10-01    -231.645688
1964-11-01     930.074592
1964-12-01    2176.794872
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trend = 'cuadratic').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01   -578.005495
1964-02-01   -164.897602
1964-03-01    302.732767
1964-04-01    481.885614
1964-05-01    748.560939
1964-06-01    708.758741
1964-07-01   -346.520979
1964-08-01   -889.278222
1964-09-01   -823.512987
1964-10-01   -260.225275
1964-11-01    215.584915
1964-12-01    604.917582
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trend = 'cubic').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01    196.725275
1964-02-01   -235.327672
1964-03-01   -190.277722
1964-04-01   -105.031635
1964-05-01    302.503830
1964-06-01    544.421911
1964-07-01   -182.184149
1964-08-01   -443.221112
1964-09-01   -236.595738
1964-10-01    232.785215
1964-11-01    286.014985
1964-12-01   -169.813187
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trend = 'diff1').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01       0
1964-02-01    -143
1964-03-01      83
1964-04-01     -34
1964-05-01     225
1964-06-01      90
1964-07-01    -754
1964-08-01     -70
1964-09-01     710
1964-10-01    1379
1964-11-01    1463
1964-12-01    1548
dtype: int64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(trend = 'diff2').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01       0
1964-02-01       0
1964-03-01     -60
1964-04-01      49
1964-05-01     191
1964-06-01     315
1964-07-01    -664
1964-08-01    -824
1964-09-01     640
1964-10-01    2089
1964-11-01    2842
1964-12-01    3011
dtype: int64


Removing seasonality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(seasonal = 'poly2').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01   -578.005495
1964-02-01   -164.897602
1964-03-01    302.732767
1964-04-01    481.885614
1964-05-01    748.560939
1964-06-01    708.758741
1964-07-01   -346.520979
1964-08-01   -889.278222
1964-09-01   -823.512987
1964-10-01   -260.225275
1964-11-01    215.584915
1964-12-01    604.917582
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> transformed_ts = Transformer(seasonal = 'diff').fit(ts = ts)
>>> transformed_ts
Month
1964-01-01    0
1964-02-01    0
1964-03-01    0
1964-04-01    0
1964-05-01    0
1964-06-01    0
1964-07-01    0
1964-08-01    0
1964-09-01    0
1964-10-01    0
1964-11-01    0
1964-12-01    0
dtype: int64

Restore seasonality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(seasonal = 'poly2')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(seasonal = 'diff')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815
1964-02-01    2672
1964-03-01    2755
1964-04-01    2721
1964-05-01    2946
1964-06-01    3036
1964-07-01    2282
1964-08-01    2212
1964-09-01    2922
1964-10-01    4301
1964-11-01    5764
1964-12-01    7312
dtype: int64

inverse_transform trending
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trend = 'linear')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trend = 'cuadratic')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trend = 'cuadratic')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trend = 'cubic')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trend = 'diff1')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815
1964-02-01    2672
1964-03-01    2755
1964-04-01    2721
1964-05-01    2946
1964-06-01    3036
1964-07-01    2282
1964-08-01    2212
1964-09-01    2922
1964-10-01    4301
1964-11-01    5764
1964-12-01    7312
dtype: int64


>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trend = 'diff2')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815
1964-02-01    2672
1964-03-01    2755
1964-04-01    2721
1964-05-01    2946
1964-06-01    3036
1964-07-01    2282
1964-08-01    2212
1964-09-01    2922
1964-10-01    4301
1964-11-01    5764
1964-12-01    7312
dtype: int64

inverse_transform transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trans = 'log')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
Name: Perrin, dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trans = 'log10')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
Name: Perrin, dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trans = 'sqrt')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trans = 'cbrt')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.0
1964-02-01    2672.0
1964-03-01    2755.0
1964-04-01    2721.0
1964-05-01    2946.0
1964-06-01    3036.0
1964-07-01    2282.0
1964-08-01    2212.0
1964-09-01    2922.0
1964-10-01    4301.0
1964-11-01    5764.0
1964-12-01    7312.0
dtype: float64

>>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
>>> mytransform = Transformer(trans = 'boxcox')
>>> transformed = mytransform.fit(ts)
>>> original = mytransform.inverse_transform(transformed)
>>> original
Month
1964-01-01    2815.000000
1964-02-01    2671.999999
1964-03-01    2754.999999
1964-04-01    2721.000001
1964-05-01    2946.000000
1964-06-01    3036.000001
1964-07-01    2282.000000
1964-08-01    2212.000001
1964-09-01    2921.999999
1964-10-01    4300.999998
1964-11-01    5763.999989
1964-12-01    7311.999999
Name: Perrin, dtype: float64

"""

from skfore.extras import *

import pandas
import numpy
import scipy
import sklearn

from sklearn import linear_model

class Transformer():
    """ Class to transform the series

    Args:
        trans (log, log10, sqrt, cbrt, boxcox): Transformation to apply
        trend (linear, cuadratic, cubic, diff1, diff2): Trend to apply
        seasonal (poly2, diff): Seasonality to apply

    """

    def __init__(self, trans = None, trend = None, seasonal = None):
        self.trans = trans
        self.trend = trend
        self.seasonal = seasonal
        self.ts = None

        """ Frequency integer of the transformed time series """
        self.intfrq = None

        """ Time series after transformation and inverse_transform """
        self.residuals = None
        self.original = None

        """ Transformation values to inverse_transform series """
        self.fitting = None
        self.diff = None
        self.model = None

        """ Box Cox transformation lambda (if necessary) """
        self.lmbda = None


    def __repr__(self):
        return 'Transformer(trans = ' + str(self.trans) + ', trend = ' + str(self.trend) + ', seasonal = ' + str(self.seasonal) +')'
        

    def fit(self, ts):
        """ Return the transformed series

        Args:
            ts: Time series to apply transformation
        """

        self.ts = ts

        # Get frequency integer
        self.intfrq = get_frequency(ts)

        # Transform 
        if (self.trans == 'log'):
            ts_trans = numpy.log(ts)
        elif (self.trans == 'log10'):
            ts_trans = numpy.log10(ts)
        elif (self.trans == 'sqrt'):
            ts_trans = numpy.sqrt(ts)
        elif (self.trans == 'cbrt'):
            ts_trans = numpy.cbrt(ts)
        elif (self.trans == 'boxcox'):
            bc, lmb = scipy.stats.boxcox(ts)
            self.lmbda = lmb
            ts_trans = pandas.Series((v for v in bc), index = ts.index, name = ts.name)
        elif (self.trans == None):
            ts_trans = ts
        else:
            message_trans = 'Invalid transformation value: ' + self.trans
            raise ValueError(message_trans)


        # Removing trend
        if (self.trend == 'linear'):
            X = ts_trans.index.factorize()[0].reshape(-1,1)
            y = ts_trans
            model = sklearn.linear_model.LinearRegression()
            fitting = model.fit(X, y)
            self.fitting = fitting
            trend = pandas.Series(fitting.predict(X), index = y.index)
            ts_trend = y.subtract(trend)
        elif (self.trend == 'cuadratic'):
            X = ts_trans.index.factorize()[0].reshape(-1,1)
            y = ts_trans
            model = sklearn.preprocessing.PolynomialFeatures(degree=2)
            self.model = model
            X_ = model.fit(X)
            model = linear_model.LinearRegression()
            fitting = model.fit(X_, y)
            self.fitting = fitting
            trend = fitting.predict(X_)
            ts_trend = y.subtract(trend)
        elif (self.trend == 'cubic'):
            X = ts_trans.index.factorize()[0].reshape(-1,1)
            y = ts_trans
            model = sklearn.preprocessing.PolynomialFeatures(degree=3)
            self.model = model
            X_ = model.fit(X)
            model = linear_model.LinearRegression()
            fitting = model.fit(X_, y)
            self.fitting = fitting
            trend = fitting.predict(X_)
            ts_trend = y.subtract(trend)
        elif (self.trend == 'diff1'):
            y = ts_trans
            diff = list()
            diff.append(0)
            self.diff = list()
            self.diff.append(y[0])
            for i in range(1, len(y)):
                value = y[i] - y[i-1]
                diff.append(value)
            trend = diff
            detrended = pandas.Series((v for v in trend), index = ts_trans.index)
            ts_trend = detrended
        elif (self.trend == 'diff2'):
            y = ts_trans
            diff = list()
            diff.append(0)
            diff.append(0)
            self.diff = list()
            self.diff.append(y[0])
            self.diff.append(y[1])
            for i in range(2, len(y)):
                value = y[i] - y[i - 2]
                diff.append(value)
            trend = diff
            detrended = pandas.Series((v for v in trend), index = ts_trans.index)
            ts_trend = detrended
        elif (self.trend == None):
            ts_trend = ts_trans
            trend = [0 for i in range(0, len(ts_trans))]
        else:
            message_trend = 'Invalid trending value: ' + self.trend
            raise ValueError(message_trend)

        # Removing seasonality
        if (self.seasonal == 'poly2'):
            X = ts_trend.index.factorize()[0].reshape(-1,1)
            X = X%self.intfrq
            y = ts_trend
            model = sklearn.preprocessing.PolynomialFeatures(degree=2)
            self.model = model
            X_ = model.fit(X)
            model = linear_model.LinearRegression()
            fitting = model.fit(X_, y)
            seasonality = fitting.predict(X_)
            deseasonal = pandas.Series((v for v in seasonality), index = ts_trend.index)
            ts_seasonal = y.subtract(deseasonal)
        elif (self.seasonal == 'diff'):
            y = ts_trend
            diff = list()
            self.diff = list()
            for j in range(self.intfrq):
                diff.append(0)
                self.diff.append(y[j])
            for i in range(self.intfrq, len(y)):
                value = y[i] - y[i - self.intfrq]
                diff.append(value)
            seasonality = diff
            deseasonal = pandas.Series((v for v in seasonality), index = ts_trend.index)
            ts_seasonal = deseasonal
        elif (self.seasonal == None):
            ts_seasonal = ts_trend
            seasonality = [0 for i in range(0, len(ts_trend))]
        else:
            message_seasonal = 'Invalid seasonal value: ' + self.seasonal
            raise ValueError(message_seasonal)

        if (self.seasonal == 'poly2' or self.trans == 'linear' or self.trend == 'cuadratic' or self.trend == 'cubic'):
            self.fitting = fitting

        self.residuals = ts_seasonal

        return self.residuals


    def inverse_transform(self, ts):
        """ inverse_transform series to its original values

        Args:
            ts: Time series to inverse_transform
        """

        # inverse_transform seasonality
        if (self.seasonal == 'poly2'):
            X = ts.index.factorize()[0].reshape(-1,1)
            X = X%self.intfrq
            X_ = self.model.fit(X)
            seasonality = self.fitting.predict(X_)
            ts_deseasonal = [ts[i] + seasonality[i] for i in range(len(ts))]
            ts_deseasonal = pandas.Series((v for v in ts_deseasonal), index = ts.index)
        elif (self.seasonal == 'diff'):
            len_forecast = len(ts)
            #ts = self.ts.append(ts)
            ts_deseasonal = list()
            for j in range(0, self.intfrq):
                ts_deseasonal.append(self.diff[j])
            for i in range(self.intfrq, len(ts)):
                value = ts[i] + ts_deseasonal[i-self.intfrq]
                ts_deseasonal.append(value)
            ts_deseasonal = pandas.Series((v for v in ts_deseasonal), index = ts.index)
            ts_deseasonal = ts_deseasonal[-len_forecast:]
        else:
            ts_deseasonal = ts

        # inverse_transform trending 
        if (self.trend == 'linear'):
            X = ts.index.factorize()[0].reshape(-1,1)
            trending = self.fitting.predict(X)
            ts_detrend = [ts_deseasonal[i] + trending[i] for i in range(len(ts_deseasonal))]
            ts_detrend = pandas.Series((v for v in ts_detrend), index = ts_deseasonal.index)
        elif (self.trend == 'cuadratic' or self.trend == 'cubic'):
            X = ts.index.factorize()[0].reshape(-1,1)
            X_ = self.model.fit(X)
            trending = self.fitting.predict(X_)
            ts_detrend = [ts_deseasonal[i] + trending[i] for i in range(len(ts_deseasonal))]
            ts_detrend = pandas.Series((v for v in ts_detrend), index = ts_deseasonal.index)
        elif (self.trend == 'diff1'):
            ts_detrend = list()
            ts_detrend.append(self.diff[0])
            for i in range(1,len(ts_deseasonal)):
                value = ts_deseasonal[i] + ts_detrend[i-1]
                ts_detrend.append(value)
            ts_detrend = pandas.Series((v for v in ts_detrend), index = ts_deseasonal.index)
        elif (self.trend == 'diff2'):
            ts_detrend = list()
            ts_detrend.append(self.diff[0])
            ts_detrend.append(self.diff[1])
            for i in range(2,len(ts_deseasonal)):
                value = ts_deseasonal[i] + ts_detrend[i-2]
                ts_detrend.append(value)
            ts_detrend = pandas.Series((v for v in ts_detrend), index = ts_deseasonal.index)
        else:
            ts_detrend = ts_deseasonal

        # inverse_transform transformation
        if (self.trans == 'log'):
            ts_detrans = numpy.exp(ts_detrend)
        elif (self.trans == 'log10'):
            ts_detrans = scipy.special.exp10(ts_detrend)
        elif (self.trans == 'sqrt'):
            ts_detrans = [ts_detrend[i]**2 for i in range(len(ts_detrend))]
            ts_detrans = pandas.Series((v for v in ts_detrans), index = ts_detrend.index)
        elif (self.trans == 'cbrt'):
            ts_detrans = [ts_detrend[i]**3 for i in range(len(ts_detrend))]
            ts_detrans = pandas.Series((v for v in ts_detrans), index = ts_detrend.index)
        elif (self.trans == 'boxcox'):
            ts_detrans = scipy.special.inv_boxcox(ts_detrend, self.lmbda)
        else:
            ts_detrans = ts_detrend

        self.original = ts_detrans

        return self.original


if __name__ == "__main__":
    import doctest
    doctest.testmod()
