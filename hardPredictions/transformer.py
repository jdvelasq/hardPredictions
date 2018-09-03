"""
Transformer
Time series' transformation module

"""

import hardPredictions.extras

import pandas
import numpy
import scipy
import sklearn

from sklearn import linear_model

class transformer():
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
        
        """ Frequency integer od the transformed time series """
        self.intfrq = None
        
        """ Time series after transformation and restore """
        self.residuals = None
        self.original = None
        
        """ Transformation values to restore series """
        self.fitting = None
        self.diff = None
        self.model = None
        
        """ Box Cox transformation lambda (if necessary) """
        self.lmbda = None
        
        
    def __repr__(self):
        return self.ts.__repr__()
    
    def fit_transform(self, ts):
        """ Return the transformed series

        Args:
            ts: Time series to apply transformation         
        """       
        
        
        """ Get frequency integer """
        self.intfrq = hardPredictions.extras.get_frequency(ts)          
        
        
        """ Applying transformation
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trans = 'log').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trans = 'log10').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trans = 'sqrt').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trans = 'cbrt').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trans = 'boxcox').fit_transform(ts = ts)
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
        
        """
        
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
            
        
        """ Removing trend 
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trend = 'linear').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trend = 'cuadratic').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trend = 'cubic').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trend = 'diff1').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(trend = 'diff2').fit_transform(ts = ts)
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
        
        """
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
            X_ = model.fit_transform(X)
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
            X_ = model.fit_transform(X)
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
        
        """ Removing seasonality
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(seasonal = 'poly2').fit_transform(ts = ts)
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
        
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> transformed_ts = transformer(seasonal = 'diff').fit_transform(ts = ts)
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
    
        """
        if (self.seasonal == 'poly2'):
            X = ts_trend.index.factorize()[0].reshape(-1,1)
            X = X%self.intfrq       
            y = ts_trend   
            model = sklearn.preprocessing.PolynomialFeatures(degree=2)
            self.model = model
            X_ = model.fit_transform(X)
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
   
    
    def restore(self, ts):
        """ Restore series to its original values 
        
        Args:
            ts: Time series to restore         
        """  
        
        """ Restore seasonality
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(seasonal = 'poly2')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    3893.956201
        1965-02-01    2889.402694
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(seasonal = 'diff')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
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
        1965-01-01    2815.0
        1965-02-01    2672.0
        dtype: float64

        """        
        if (self.seasonal == 'poly2'):
            X = ts.index.factorize()[0].reshape(-1,1)
            X = X%self.intfrq
            X_ = self.model.fit_transform(X)
            seasonality = self.fitting.predict(X_)
            ts_deseasonal = [ts[i] + seasonality[i] for i in range(len(ts))]
            ts_deseasonal = pandas.Series((v for v in ts_deseasonal), index = ts.index)
        elif (self.seasonal == 'diff'):
            ts_deseasonal = list()
            for j in range(0, self.intfrq):
                ts_deseasonal.append(self.diff[j])
            for i in range(self.intfrq, len(ts)):
                value = ts[i] + ts_deseasonal[i-self.intfrq]
                ts_deseasonal.append(value)
            ts_deseasonal = pandas.Series((v for v in ts_deseasonal), index = ts.index)
        else:
            ts_deseasonal = ts
            
        """ Restore trending
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trend = 'linear')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    8484.284463
        1965-02-01    9098.510083
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trend = 'cuadratic')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    8538.178375
        1965-02-01    9591.355561
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trend = 'cuadratic')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    8538.178375
        1965-02-01    9591.355561
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trend = 'cubic')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01     2815.000000
        1964-02-01     2672.000000
        1964-03-01     2755.000000
        1964-04-01     2721.000000
        1964-05-01     2946.000000
        1964-06-01     3036.000000
        1964-07-01     2282.000000
        1964-08-01     2212.000000
        1964-09-01     2922.000000
        1964-10-01     4301.000000
        1964-11-01     5764.000000
        1964-12-01     7312.000000
        1965-01-01     9868.705164
        1965-02-01    13524.981509
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trend = 'diff1')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    8712.463721
        1965-02-01    9914.728326
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trend = 'diff2')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01     2815.000000
        1964-02-01     2672.000000
        1964-03-01     2755.000000
        1964-04-01     2721.000000
        1964-05-01     2946.000000
        1964-06-01     3036.000000
        1964-07-01     2282.000000
        1964-08-01     2212.000000
        1964-09-01     2922.000000
        1964-10-01     4301.000000
        1964-11-01     5764.000000
        1964-12-01     7312.000000
        1965-01-01     8989.726850
        1965-02-01    11294.037281
        dtype: float64
        
        """         
        if (self.trend == 'linear'):
            X = ts.index.factorize()[0].reshape(-1,1)
            trending = self.fitting.predict(X)
            ts_detrend = [ts_deseasonal[i] + trending[i] for i in range(len(ts_deseasonal))]
            ts_detrend = pandas.Series((v for v in ts_detrend), index = ts_deseasonal.index)
        elif (self.trend == 'cuadratic' or self.trend == 'cubic'):
            X = ts.index.factorize()[0].reshape(-1,1)
            X_ = self.model.fit_transform(X)
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
       
        """ Restore transformation
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trans = 'log')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    3558.758272
        1965-02-01    3449.753767
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trans = 'log10')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    3558.753355
        1965-02-01    3449.752959
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trans = 'sqrt')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    4472.882806
        1965-02-01    3533.515931
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trans = 'cbrt')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
        1964-01-01    2815.000000
        1964-02-01    2672.000000
        1964-03-01    2755.000000
        1964-04-01    2721.000000
        1964-05-01    2946.000000
        1964-06-01    3036.000000
        1964-07-01    2282.000000
        1964-08-01    2212.000000
        1964-09-01    2922.000000
        1964-10-01    4301.000000
        1964-11-01    5764.000000
        1964-12-01    7312.000000
        1965-01-01    4023.074533
        1965-02-01    3513.214894
        dtype: float64
        
        >>> from AR import AR
        >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
        >>> mytransform = transformer(trans = 'boxcox')
        >>> transformed = mytransform.fit_transform(ts)
        >>> model = AR(p = 3)
        >>> model = model.fit(transformed)
        >>> ahead = model.forecast(transformed, periods = 2)
        >>> original = mytransform.restore(ahead)
        >>> original
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
        1965-01-01     518.655992
        1965-02-01     518.679431
        dtype: float64
        
        """
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
