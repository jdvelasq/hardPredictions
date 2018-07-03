"""
Base Model
------------------

Base structure for creation of new models

Methods:
    calc_error: Estimates error accordin to SciKit's regression metrics
    filter_ts: Returns model's residuals

"""

from hardPredictions.hardPredictions import series_viewer
import sklearn

class base_model():
    
    def __init__(self):
        self.residuals = None        
        self.test()
        
    def test(self):
        """ Raises error if there are not any of the necessary methods defined """
        
        if (not "predict" in dir(self)):
            raise ValueError('Method "predict" has not been defined')
        if (not "fit" in dir(self)):
            raise ValueError('Method "fit" has not been defined')
        if (not "forecast" in dir(self)):
            raise ValueError('Method "forecast" has not been defined')    
        
    
    #def __repr__(self):
    #    return self.ts.__repr__()

    def calc_error(self, ts, error_type = None):
        """ Estimates error according to SciKit's regression metrics
        
        Args:
            ts: Time series to estimate the model
            error_type (mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score, explained_variance_score): Error type
        
        """
        y_estimated = self.predict(ts)
        y_real = ts
        
        if (error_type == None or error_type == 'mean_squared_error'):
            error = sklearn.metrics.mean_squared_error(y_real, y_estimated)
        elif (error_type == 'mean_absolute_error'):
            error = sklearn.metrics.mean_absolute_error(y_real, y_estimated)
        #elif (error_type == 'mean_squared_log_error'):
        #    error = sklearn.metrics.mean_squared_log_error(y_real, y_estimated)
        elif (error_type == 'median_absolute_error'):
            error = sklearn.metrics.median_absolute_error(y_real, y_estimated)
        elif (error_type == 'r2_score'):
            error = sklearn.metrics.r2_score(y_real, y_estimated)
        elif (error_type == 'explained_variance_score'):
            error = sklearn.metrics.explained_variance_score(y_real, y_estimated)
        else:
            message_error_type = 'Invalid error type: ' + error_type
            raise ValueError(message_error_type)        
        
        return error

    
    def filter_ts(self, ts):
        """ Returns model's residuals
        
        Args:
            ts: Time series to estimate residuals
            
        """
        prediction = self.predict(ts)
        residuals = ts.subtract(prediction)
        return residuals            
   
    
    def set_residuals(self, residuals):
        self.residuals = series_viewer(residuals)    
        
    
    """ Residuals analysis """
    def time_plot(self):
        self.residuals.time_plot()
        
    def ACF_plot(self):
        self.residuals.ACF_plot()
    
    def PACF_plot(self):
        self.residuals.PACF_plot()
        
    def qq_plot(self):
        self.residuals.qq_plot()
        
    def density_plot(self):
        self.residuals.density_plot()
        
    def histogram(self):
        self.residuals.histogram()
    
    def normality(self):
        self.residuals.normality()   
