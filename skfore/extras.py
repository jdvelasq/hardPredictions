"""
ExtraFunctions
skfore

"""

import pandas
import math


def aic(k, error):
    """ Akaike Information Criteria """
        
    AIC = 2*k + 2*math.log(error)        
    return AIC

def bic(n, k, error):
    """ Bayesian Information Criteria """
        
    BIC = n*math.log(error/n) + k*math.log(n)       
    return BIC        


def get_frequency(ts):
    """ Find a series' frequency integer
        
    >>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
    >>> int_frq = get_frequency(ts)
    >>> int_frq
    12
        
    """
    try:
        frq = pandas.infer_freq(ts.index)
    except:
        frq = 'A'  
    int_frq = len(pandas.date_range(pandas.datetime(2017, 1, 1), pandas.datetime(2017, 12, 31), freq = frq))
    return int_frq

def get_pandas_frequency(ts):
    """ Find a series' frequency integer
        
    >>> ts = pandas.Series.from_csv('datasets/champagne_short.csv', index_col = 0, header = 0)
    >>> int_frq = get_frequency(ts)
    >>> int_frq
    12
        
    """
    try:
        frq = pandas.infer_freq(ts.index)
    except:
        frq = 'A'  
    return frq


def add_next_date(ts, value):
    """ Assigns a value to the next date in a series
    
    Args:
        ts: Time series to which the value will be added
        value: Value to add
    
    """
    if type(ts.index) == pandas.core.indexes.numeric.Int64Index:
        next_date = [ts.index[-1], ts.index[-1] + 1]
    else:
        next_date = pandas.date_range(ts.index[-1], periods=2, freq=get_pandas_frequency(ts))
    next_ts = pandas.Series(value, index = next_date)
    next_ts = next_ts.drop(next_ts.index[0])
    ts_forecast = ts.append(next_ts)
    return ts_forecast



#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()  