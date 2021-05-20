"""
COVID19 Colombia Case
"""

import pandas
from os.path import join, dirname

module_path = dirname(__file__) 
covid = pandas.Series.from_csv(join(module_path, "datasets/WWWusage.csv"), index_col = 0, header = 0)