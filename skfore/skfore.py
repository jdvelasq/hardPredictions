"""
skfore
A library in Python for time series forecasting

"""


import scipy
import statsmodels
import tabulate


from tabulate import tabulate
from scipy import stats
from statsmodels import api


""" Time series' and model's diagnostics """
class series_viewer():
    def __init__(self, ts=None):
        self.ts = ts

    def describe(self):
        td = self.ts.describe()
        return td

    def plot(self):
        tp = self.ts.plot()
        return tp

    """ Histogram of frequencies """
    def histogram(self):
        hist = self.ts.hist()
        return hist

    def density_plot(self):
        den_plot = self.ts.plot(kind='kde')
        return den_plot

    def ACF_plot(self):
        acf = statsmodels.graphics.tsaplots.plot_acf(self.ts)
        return acf

    def PACF_plot(self):
        pacf = statsmodels.graphics.tsaplots.plot_pacf(self.ts)
        return pacf

    def qq_plot(self):
        qq_plot = statsmodels.graphics.gofplots.qqplot(self.ts)
        return qq_plot


    """ Jarque Bera normality test """
    def normality(self):
        jb = scipy.stats.jarque_bera(self.ts)
        jb_value = jb[0]
        p_value = jb[1]
        return (tabulate([['Test statistics value', jb_value], ['p value', p_value]], headers = ['Jarque Bera normality test'], tablefmt = 'html'))     
