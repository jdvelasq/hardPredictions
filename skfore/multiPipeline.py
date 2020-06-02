"""

skfore Pipeline for multiple time series


"""

import Pipeline

class multiPipeline():
    
    def __init__(self, files = None, transformation = None, model = None):
        self.files = files
        self.transformation = transformation
        self.model = model
        
    def fit(self):
        errors = list()
        for i in self.files:
            file_errors = Pipeline.fit(i, self.transformation, self.model)
            errors.append(file_errors)
        return errors
        