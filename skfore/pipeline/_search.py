"""

"""

from sklearn.model_selection import ParameterGrid
import copy


class GridSearchCV():
    
    def __init__(self, model, parameters, error_function = None):
        self.initial_model = model
        self.model = model
        self.parameters = parameters
        self.param_grid = ParameterGrid(parameters)
        
        self.cv_results_ = list()
        self.best_score_ = None
        self.best_params_ = None
        self.best_model_ = None
        
    def __repr__(self):
        return 'GridSearchCV(model = ' + str(self.initial_model) + ', parameters = ' + str(self.parameters) +')'
    
    def fit(self, ts, tst = None, **fit_params):

            
        try:
            
            for parameters in self.param_grid:
                model = copy.deepcopy(self.model)
                model.__init__()
                model = model.update(parameters)
                model.fit(ts, **fit_params)
                i_error = model.calc_error(tst)
                
                params = parameters
                params['score'] = i_error
                self.cv_results_.append(params)
                
                if self.best_score_ == None or i_error < self.best_score_:
                    self.best_score_ = i_error
                    self.best_params_ = parameters
                    self.best_model_ = model
                
                print(i_error)
        
        except:
             
            for parameters in self.param_grid:
                model = copy.deepcopy(self.model)
                model.__init__()
                model = model.update(parameters)
                model.fit(ts, **fit_params)
                i_error = model.calc_error(ts)
                
                params = parameters
                params['score'] = i_error
                self.cv_results_.append(params)
                
                if self.best_score_ == None or i_error < self.best_score_:
                    self.best_score_ = i_error
                    self.best_params_ = parameters
                    self.best_model_ = model
                
                print(i_error)
                    
            
            

        return self

    