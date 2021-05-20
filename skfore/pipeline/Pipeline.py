"""

skfore Pipeline

"""

import skfore.preprocessing.transformer

class Pipeline():

    def __init__(self, ts, transformation = None, model = None, error_function = None):
        self.transformation = transformation
        self.model = model
        self.ts = ts
        self.error_function = error_function


    def fit(self):
        error_list = list()
        for trans_i in self.transformation:
            for model_i in self.model:
                model = model_i
                trans = trans_i
                transformed = trans.fit_transform(self.ts)
                model_s = model.fit(transformed)
                error = model.calc_error(transformed, self.error_function)
                print(trans)
                print(model)
                print(error)
                error_list.append(error)
        return error_list
