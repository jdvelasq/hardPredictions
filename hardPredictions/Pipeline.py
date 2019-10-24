"""

hardPredictions Pipeline

"""

import transformer

class Pipeline():
    
    def __init__(self, ts, transformation = None, model = None):
        self.transformation = transformation
        self.model = model
        self.ts = ts
        
    
    def fit(self):        
        for trans_i in self.transformation:
            for model_i in self.model:
                model = model_i
                trans = trans_i
                transformed = trans.fit_transform(self.ts)
                model_s = model.fit(transformed)
                error = model.calc_error(transformed)
                print(trans)
                print(model)
                print(error)
        return error
            
        
    
        
        

