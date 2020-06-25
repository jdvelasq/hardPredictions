"""
Datasets

"""
import pandas
from os.path import join, dirname

def load_paper(): 
    
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/paper.csv"), index_col = 0, header = 0)
    return fdata