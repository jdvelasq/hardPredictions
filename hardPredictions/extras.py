"""
ExtraFunctions
pytimeseries

"""

import pandas


def get_frequency(ts):
    """ Find a series' frequency integer
        
    >>> ts = pandas.Series.from_csv('champagne_short.csv', index_col = 0, header = 0)
    >>> int_frq = get_frequency(ts)
    >>> int_frq
    12
        
    """
    
    frq = pandas.infer_freq(ts.index)
    int_frq = len(pandas.date_range(pandas.datetime(2017, 1, 1), pandas.datetime(2017, 12, 31), freq = frq))
    return int_frq


def add_next_date(ts, value):
    """ Assigns a value to the next date in a series
    
    Args:
        ts: Time series to which the value will be added
        value: Value to add
    
    """
    
    next_date = pandas.date_range(ts.index[-1], periods=2, freq=pandas.infer_freq(ts.index))
    next_ts = pandas.Series(value, index = next_date)
    next_ts = next_ts.drop(ts.index[-1])
    ts_forecast = ts.append(next_ts)
    return ts_forecast



if __name__ == "__main__":
    import doctest
    doctest.testmod()  