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

# Preliminary analysis 
# Seven or above rating is very good quality
# FIve or six to have average quality
# Less than five qualtiy to be poor

n_wines = data.shape[0]

# Number of wines with quality rating above six
quality_above_six = data.loc[(data['quality'] > 6)]
num_above_six = quality_above_six.shape[0]

# Number of wines with quality rating below five
quality_below_five = data.loc[(data['quality'] < 5)]
num_below_five = quality_below_five.shape[0]

# Number of wines with quality rating between five to six
quality_between_five_and_six = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6 )]
numb_between_five_and_six = quality_between_five_and_six.shape[0]

# Figure out the percentage of wines with quality rating above 6
percentage_of_wines_above_six = num_above_six*100/n_wines

# Results
print("Total number of wine data: {}".format(n_wines))
print("Wines with rating 7 and above: {}".format(num_above_six))
print("Wines with rating less than 5: {}".format(num_below_five))
print("Wines with rating 5 and 6: {}".format(numb_between_five_and_six))
print("Percentage of wines with quality 7 and above: {:.2f}%".format(percentage_of_wines_above_six))
# Additonal data analysis
display(np.round(data.describe()))
# Visualised features of the original data
vs.distribution(data, "quality")