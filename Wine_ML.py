import numpy as np
import pandas as pd
from time import time
# Allows the use of the display() for displaying DataFrames
# Module that displays data structures in a better format
from IPython.display import display

import matplotlib.pyplot as plt
# Makes prettier graphs and uses matplotlibs to provide better visualization
import seaborn as sns

# Import supplementary visualization code visuals.py 
import visuals as vs

# Load the red wines dataset
data = pd.read_csv("data/winequality-red.csv", sep= ';')

# Displaying the first five records
display(data.head(n = 5))

# Check if any columns have missing information
data.isnull().any()

# To Get more information on the dataset
data.info()