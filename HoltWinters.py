"""

Holt Winters Model

"""

from hardPredictions.base_model import base_model

import numpy
import scipy
import pandas
import hardPredictions.extras
import random
from sklearn import linear_model

class HoltWinters(base_model):
    """ """
    
    def __init__(self, alpha=None, beta=None, gamma=None, seasonal='additive'):
        
        #""" Checks parameters are between 0 and 1 """
        #if alpha != None and (alpha < 0 or alpha > 1):
        #    alpha_limits = 'Error: parameter "alpha" must be between 0 and 1'
        #    raise ValueError(alpha_limits)
            
        #if beta != None and (beta < 0 or beta > 1):
        #    beta_limits = 'Error: parameter "beta" must be between 0 and 1'
        #    raise ValueError(beta_limits)
            
        #if gamma != None and (alpha < 0 or alpha > 1):
        #    gamma_limits = 'Error: parameter "gamma" must be between 0 and 1'
        #    raise ValueError(gamma_limits)
            
        """ Checks parameters """
        if seasonal != 'additive' or seasonal != 'multiplicative':
            seasonal_error = 'Error: Invalid seasonal value: ' + self.seasonal
            raise ValueError(seasonal_error)
        
        if self.alpha == False:
            message_alpha = 'Error: parameter "alpha" cannot be False'
            raise ValueError(message_alpha)
        
        if self.alpha != False and self.beta == False and self.gamma != False:
            message_beta = 'Error: parameter "beta" has not been defined'
            raise ValueError(message_beta)          
        
            
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal = seasonal
        
    def params2vector(self):
        params = list()
        if self.beta == False and self.gamma == False:
            params.append(self.alpha)        
        elif self.gamma == False:
            params.append(self.alpha)
            params.append(self.beta)        
        else:
            params.append(self.alpha)
            params.append(self.beta)
            params.append(self.gamma)
        
        return params
    
    def vector2params(self, vector):        
        if self.beta == False and self.gamma == False:
            self.alpha = vector[0]        
        elif self.gamma == False:
            self.alpha = vector[0]
            self.beta = vector[1]            
        else:
            self.alpha = vector[0]
            self.beta = vector[1]
            self.gamma = vector[2]
            
        return self
    
    def predict(self, ts):
        
        def additive(self, ts):
            
            def initial_trend(series, slen):
                sum = 0.0
                for i in range(slen):
                    sum += float(series[i+slen] - series[i]) / slen
                return sum / slen
            
            def initial_seasonal_components(series, slen):
                seasonals = {}
                season_averages = []
                n_seasons = int(len(series)/slen)
        
                for j in range(n_seasons):
                    season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
                
                for i in range(slen):
                    sum_of_vals_over_avg = 0.0
                    for j in range(n_seasons):
                        sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
                    seasonals[i] = sum_of_vals_over_avg/n_seasons
                
                return seasonals
            
            slen = len(ts)
            result = []
            seasonals = initial_seasonal_components(ts, slen)
            
            for i in range(len(ts)):
                if i == 0: 
                    smooth = ts[0]
                    trend = initial_trend(ts, slen)
                    result.append(ts[0])
                    continue
                if i >= len(ts): 
                    m = i - len(ts) + 1
                    result.append((smooth + m*trend) + seasonals[i%slen])
                else:
                    val = ts[i]
                    last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%slen]) + (1-self.alpha)*(smooth+trend)
                    trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                    seasonals[i%slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%slen]
                    result.append(smooth+trend+seasonals[i%slen])
            
            return result
        
        def multiplicative(self, ts):
            
            def initial_trend(series, slen):
                sum = 0.0
                for i in range(slen):
                    sum += float(series[i+slen] - series[i]) / slen
                return sum / slen
            
            def initial_seasonal_components(series, slen):
                seasonals = {}
                season_averages = []
                n_seasons = int(len(series)/slen)
        
                for j in range(n_seasons):
                    season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
                
                for i in range(slen):
                    sum_of_vals_over_avg = 0.0
                    for j in range(n_seasons):
                        sum_of_vals_over_avg += series[slen*j+i]/season_averages[j]
                    seasonals[i] = sum_of_vals_over_avg/n_seasons
                
                return seasonals
            
            slen = len(ts)
            result = []
            seasonals = initial_seasonal_components(ts, slen)
            
            for i in range(len(ts)):
                if i == 0: 
                    smooth = ts[0]
                    trend = initial_trend(ts, slen)
                    result.append(ts[0])
                    continue
                if i >= len(ts): 
                    m = i - len(ts) + 1
                    result.append((smooth + m*trend) + seasonals[i%slen])
                else:
                    val = ts[i]
                    last_smooth, smooth = smooth, self.alpha*(val/seasonals[i%slen]) + (1-self.alpha)*(smooth+trend)
                    trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                    seasonals[i%slen] = self.gamma*(val/smooth) + (1-self.gamma)*seasonals[i%slen]
                    result.append(smooth+trend+seasonals[i%slen])
            
            return result
        
        def _alpha_beta(self, ts):
            result = [ts[0]]
            for n in range(1, len(ts)+1):
                if n == 1:
                    level, trend = ts[0], ts[1] - ts[0]
                if n >= len(ts): 
                    value = result[-1]
                else:
                    value = ts[n]
                    last_level, level = level, self.alpha*value + (1-self.alpha)*(level+trend)
                    trend = self.beta*(level-last_level) + (1-self.beta)*trend
                    result.append(level+trend)
                
            return result
        
        def _alpha(self, ts):
            result = [ts[0]] 
            for n in range(1, len(ts)):
                result.append(self.alpha * ts[n] + (1 - self.alpha) * result[n-1])
                
            return result
            
        
        if self.alpha != None and self.beta == False and self.gamma == False:
            result = _alpha(self, ts)
        
        elif self.alpha == None and self.beta == False and self.gamma == False:
            self.fit(ts)
            result = _alpha(self, ts)
            
        elif self.alpha != None and self.beta != None and self.gamma == False:
            result = _alpha_beta(self, ts)
        
        elif self.alpha == None and self.beta == None and self.gamma == False:
            self.fit(ts)
            result = _alpha_beta(self, ts)
        
        elif self.alpha != None and self.beta != None and self.gamma != None:            
            if self.seasonal == 'multiplicative':
                result = multiplicative(self, ts)
            else:
                result = additive(self, ts)      
                
        prediction = pandas.Series((v for v in result), index = ts.index)
        return prediction 
    
    def fit(self, ts, error_type = 'mean_squared_error'):
    
        def f(x):
            self.vector2params(x)            
            return self.calc_error(ts, error_type)
        
        x0 = self.params2vector()
        optim_params = scipy.optimize.minimize(f, x0)
        self.vector2params(vector = optim_params.x)   
        self.ts = self.predict(ts)  
        
        return self

    
    def forecast(self, ts, periods):
        pass
