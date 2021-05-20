"""


skfore file reader


"""

import pandas
import glob

class file_reader():
    
    def __init__(self):
        pass
        
    def read_dir(directory = '', extension = '*'):
        if directory[0:2] != '../' and directory[0:1] != './' and directory[0] != '/':
            add_dir_1 = '../'
        else:
            add_dir_1 = ''
        if directory[-1] != '/':
            add_dir_2 = '/'
        else:
            add_dir_2 = ''
        files = glob.glob(add_dir_1 + directory + add_dir_2 + '*.' + extension)
        series = list()
        for i in range(len(files)):
            if extension == 'csv':
                x_series = pandas.read_csv(files[i], index_col = 0, header = 0)
                series.append(x_series)
            elif extension == 'txt':
                x_series = pandas.read_csv(files[i], index_col = 0, header = 0)
                series.append(x_series)
            elif extension == '*':
                x_series = pandas.read_csv(files[i], index_col = 0, header = 0)
                series.append(x_series)
            else:
                #json
                raise ValueError('Extension not recognized')
        return series
        
