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

# Scatter plot to get details about the feature set in the data.
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde')

# A heatmap of co-relations between the features
correlation = data.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin= -1, cmap= "RdBu_r")

## pH vs Fixed Acidity
# Create a new dataframe containing only pH and fixed acidity columns to visualize their co-relations
fixedAcidity_pH = data[['pH', 'fixed acidity']]
# Initialize a joint-grid with the dataframe, using seaborn library
gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, height=6)
# Draw a regression plot in the grid
gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s":10})
# Draws a distribution plot in the same grid
gridA = gridA.plot_marginals(sns.histplot)
#as fixed acidity levels increase, the pH levels drop.

## Fixed Acidity vs Citric Acid
# A new dataframe that contains only citric acid and fixed acidity columns to visualize their co-relations
fixedAcidity_citricAcid = data[['citric acid', 'fixed acidity']]
# Initialise a joint-grid with the dataframe,using the seaborn library
g = sns.JointGrid(x = "fixed acidity", y = "citric acid", data = fixedAcidity_citricAcid, height=6)
# Draw a regression plot in the grid
g = g.plot_joint(sns.regplot, scatter_kws = {"s":10})
# Draw a distribution plot in the same grid
g = g.plot_marginals(sns.histplot)


## Density vs Fixed Acidity
# A new dataframe that contains only Density and Fixed Acidity columns to visualize their co-relations
fixedAcidity_Density = data[['density', 'fixed acidity']]
# Initialise a joint-grid with the dataframe,using the seaborn library
g = sns.JointGrid(x = "fixed acidity", y = "density", data = fixedAcidity_Density, height=6)
# Draw a regression plot in the grid
g = g.plot_joint(sns.regplot, scatter_kws = {"s":10})
# Draw a distribution plot in the same grid
g = g.plot_marginals(sns.histplot)

## Volatile Acidity vs Quality
# A new dataframe that contains only Volatile Acidity and quality columns to visualize their co-relations
VolatileAcidity_Quality = data[['quality', 'volatile acidity']]
# Initialise a joint-grid with the dataframe,using the seaborn library
g = sns.JointGrid(x = "volatile acidity", y = "quality", data = VolatileAcidity_Quality, height=6)
# Draw a regression plot in the grid
g = g.plot_joint(sns.regplot, scatter_kws = {"s":10})
# Draw a distribution plot in the same grid
g = g.plot_marginals(sns.histplot)

## Using a bar plot to show relationships between discrete values
## Volatile Acidity vs Quality
fig, axs = plt.subplots(ncols = 1, figsize=(10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = VolatileAcidity_Quality, ax = axs)

plt.tight_layout()
plt.show()
plt.gcf().clear()

## Alcohol vs Quality
# A new dataframe that contains only Alcohol and quality columns to visualize their co-relations
Alcohol_Quality = data[['quality', 'alcohol']]
# Initialise a joint-grid with the dataframe,using the seaborn library
g = sns.JointGrid(x="alcohol", y="quality", data = VolatileAcidity_Quality, height=6)
# Draw a regression plot in the grid
g = g.plot_joint(sns.regplot, scatter_kws = {"s":10})
# Draw a distribution plot in the same grid
g = g.plot_marginals(sns.histplot)

# Using a bar plot to show relationships between discrete values
fig, axs = plt.subplots(ncols = 1, figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = Alcohol_Quality, ax = axs)
plt.title('Quality vs Alcohol')

plt.tight_layout()
plt.show()
plt.gcf().clear()

## Outlier Detection

''' Tukey's Method for Detecting Outliers

First the sorted data is divided into four intervals.
This is done in a way that each interval would contain about twenty five percent of the total data points.
The value at which these intervals are split are called Quartiles.

Then you subtract the third Quartile from the first Quartile to get the Interquartile Range (IQR).
That is the middle fifty percent of the data and it contains the bulk of the data.

Any data point that lies beyond one point five times the IQR would be considered as an outlier.
'''

