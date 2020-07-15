"""
Datasets

"""
import pandas
from os.path import join, dirname

""" DAN2 Series """
def load_paper():     
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/paper.csv"), index_col = 0, header = 0)
    return fdata

def load_www():     
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/WWWusage.csv"), index_col = 0, header = 0)
    return fdata

def load_passengers():     
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/passengers.csv"), index_col = 0, header = 0)
    return fdata

def load_sunspot():
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/sunspot.csv"), index_col = 0, header = 0)
    return fdata

def load_pollution():
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/pollution.csv"), index_col = 0, header = 0)
    return fdata

def load_lynx():
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/lynx.csv"), index_col = 0, header = 0)
    return fdata

""" Machine Learning Mastery Series"""
def load_champagne():
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/champagne.csv"), index_col = 0, header = 0)
    return fdata

""" Colombia Electricity Prices """
def load_contrato():
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/contrato.csv"), index_col = 0, header = 0)
    return fdata

def load_demanda():
    
    module_path = dirname(__file__) 
    fdata = pandas.Series.from_csv(join(module_path, "datasets/demanda.csv"), index_col = 0, header = 0)
    return fdata


