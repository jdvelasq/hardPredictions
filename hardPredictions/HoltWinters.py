"""

Holt Winters Model
===============================================================================

Overview
-------------------------------------------------------------------------------

This module contains Holt Winters or Exponential Smoothing model.


Examples
-------------------------------------------------------------------------------

All parameters can be optimized by choosing seasonal type: additive or
multiplicative. Additive seasonal is set by default.

>>> ts = pandas.Series.from_csv('../datasets/champagne.csv', index_col = 0, header = 0)
>>> model = HoltWinters()
>>> model
HoltWinters(alpha = None, beta = None, gamma = None, seasonal = additive)
>>> model = model.fit(ts)
>>> model
HoltWinters(alpha = 0.08779225079496673, beta = -0.03280832897112478, gamma = 0.9985667470201687, seasonal = additive)
>>> fitted_model = model.predict(ts)
>>> prediction = model.forecast(ts, periods = 2)
>>> prediction
            ci_inf  ci_sup       series
1972-10-01     NaN     NaN  6841.767847
1972-11-01     NaN     NaN  9754.197706
>>> prediction = model.forecast(ts, periods = 2, confidence_interval = 0.95)
>>> prediction
                 ci_inf       ci_sup       series
1972-10-01  6817.142357  6851.141877  6841.767847
1972-11-01  9729.312585  9763.392881  9754.197706

None parameters will be optimized even if other parameters are set:

>>> model = HoltWinters(alpha = 0.9)
>>> model
HoltWinters(alpha = 0.9, beta = None, gamma = None, seasonal = additive)
>>> model = model.fit(ts)
>>> model
HoltWinters(alpha = 0.9, beta = 0.04986683901737994, gamma = 0.2825125430030021, seasonal = additive)
>>> prediction = model.forecast(ts, periods = 2, confidence_interval = 0.95)
>>> prediction
                 ci_inf       ci_sup       series
1972-10-01  6903.954800  7075.239583  7012.853656
1972-11-01  9648.596015  9897.577641  9836.879742

Parameters can also be False if they do not want to be found:

>>> model = HoltWinters(alpha = 0.9, beta = 0.1, gamma = False)
>>> model
HoltWinters(alpha = 0.9, beta = 0.1, gamma = False, seasonal = additive)
>>> model = model.fit(ts)
>>> model
HoltWinters(alpha = 0.9, beta = 0.1, gamma = False, seasonal = additive)
>>> prediction = model.forecast(ts, periods = 2, confidence_interval = 0.95)
>>> prediction
                 ci_inf       ci_sup       series
1972-10-01  4814.564979  5812.226356  5495.803368
1972-11-01  4731.113279  6040.021961  5572.626448

"""

from hardPredictions.base_model import base_model

import numpy
import scipy
import pandas
import matplotlib
