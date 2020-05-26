"""
Datasets

"""

from os.path import join, dirname
class Bunch(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass

def load_paper(): 
    """Load and return the dynacol dataset. This dataset contains the bibliographical information about 
    publications in Scopus of the Dyna-Colombia journal, edited by Facultad de Minas, 
    Universidad Nacional de Colombia, Sede Medellin, between January, 2010 and September, 2019. 
    
    Args: None. 
    
    Returns: A dictionary.
    
    **Examples** 
    >>> from techminer.datasets import load_dynacol 
    >>> data = load_dynac """
    
    module_path = dirname(__file__) 
    with open(join(module_path, "datasets/paper.rst")) as rst_file: 
        fdescr = rst_file.read() 
    fdata = pandas.Series.read_csv(join(module_path, "datasets/paper.csv"), index_col = 0, header = 0)
    return Bunch(data=fdata, DESCR=fdescr)