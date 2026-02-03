from os import listdir
from re import search

modules = [x[:-3] for x in listdir(*__path__)
           if not "__" in x[:2] and ".py" in x[-3:]]

