"""

"""

from sklearn.model_selection import ParameterGrid

class GridSearchCV():
    
    def __init__(self, model, parameters, error_function = None):
        self.model = model
        self.param_grid = ParameterGrid(parameters)
    
    def fit(self, ts, **fit_params):
        
        error = list()
        cv_results_ = list()
        best_score_ = None
        best_params_ = None
        best_model_ = None
        for parameters in self.param_grid:
            self.model.__init__()
            model = self.model.update(parameters)
            model.fit(ts, **fit_params)
            i_error = model.calc_error(ts)
            error.append(i_error)
            
            params = parameters
            params['score'] = i_error
            cv_results_.append(params)
            
            if best_score_ == None or i_error < best_score_:
                best_score_ = i_error
                best_params_ = parameters
                best_model_ = model

        return cv_results_, best_model_, best_score_, best_params_

    