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

# Import train_test_splits
from sklearn.model_selection import train_test_split

# Import classification metrics from sklearn, fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

# Import three supervised learning classification models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import GridSearchCV, make_scorer 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

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
# Five or six to have average quality
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
g = sns.JointGrid(x="alcohol", y="quality", data=Alcohol_Quality, height=6)
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

outliers_to_remove = []
# For each feature, find the data points with the extreme high or low values
for feature in data.keys():
    # Calculate the Q1 percentage
    Q1 = np.percentile(data[feature], q = 25)
    # Calculate the Q3 percentage
    Q3 = np.percentile(data[feature], q = 75)
    # Using interquartile range to calculate the outliers
    interquartile_range = Q3 - Q1
    step = 1.5 * interquartile_range
    # Display the outliers
    print("Data points that are consdiered outliers for the feature '{}':".format(feature))
    feature_outliers = data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))]
    display(feature_outliers)
    # Add the features that are to be removed into the list
    outliers_to_remove.extend(feature_outliers.index.tolist())

# Remove Duplicates and the outliers
outliers_to_remove = list(set(outliers_to_remove))
good_data = data.drop(outliers_to_remove).reset_index(drop = True)
print(f"\nRemoved {len(outliers_to_remove)} outlier rows. New dataset shape : {good_data.shape}")

## Data Preparation
# The Regression problem is converted into a classification problem, by applying transformations.
# Then the data will be used to create a feature-set and target labels.

# Defining the splits for categories.
# 1-4 will be poor quality
# 5-6 will be average
# 7-10 will be great
bins = [1, 4, 6, 10]

# 0 for low quality
# 1 for average
# 3 for great quality
quality_labels = [ 0, 1, 2]
data['quality_catergorical'] = pd.cut(data['quality'], bins = bins, labels = quality_labels, include_lowest = True)

# Display the first two columns
display(data.head(n = 2))

# Split the data into features and target label
quality_raw = data['quality_catergorical']
features_raw = data.drop(['quality', 'quality_catergorical'], axis = 1)

# Split the features and the income data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw,
                                                    quality_raw,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


def train_predict_evaluate(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    Inputs provided:

        learner :
            the learning algorithm to be trained and predicted on
        sample_size:
            the size of samples (number) to be drawn from the training set
        X_train:
            features training set
        y_train:
            quality training set
        X_test:
            fetures testing set
        y_test:
            quality testing set
    '''

    results = {}

    '''
    Train the learner to the training data using slicing with 'sample_size'
    using the .fit(training_features[:], training_labels[:])
    '''
    # Start time of training
    start = time()
    # Model gets trained here
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    # End time of farming
    end = time()

    # Calculate the total training time
    results['train_time'] = end - start
    '''
        Get the predictions on the first three hundred training samples (X_train), and also the predictions on the test set(X_test) using .predict()
    '''
    # Start time
    start = time()
    predictions_train = learner.predict(X_train[:300])
    predictions_test = learner.predict(X_test)
    # End time   
    end = time()

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first three hundred samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # Compare accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Compute F1-Score on the first three hundret training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5, average = 'micro')

    # Compute F1-Score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5, average = 'micro')

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results



'''
Three supervised learning models are imported.
Gaussian Naive Bayes, Decision Tree and Random Forest Classifier
One Logistic Regression model is imported too.
Then the number of samples for 1%, 10%, and 100% of the training data are calculated and stored.

Then the results on the learners are stored.
'''

# Initialise the four models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(max_depth = None, random_state = None)
clf_C = RandomForestClassifier(max_depth = None, random_state = None)
clf_D = LogisticRegression()

# Calculate the number of samples for 1%, 10% and 100% of the training data

## 100% Sample
samples_100 = len(y_train)

## 10% Sample
samples_10 = int(len(y_train) * 10/100)

## 1% Sample
samples_1 = int(len(y_train) * 1/100)

# Collect the results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict_evaluate(clf, samples, X_train, y_train, X_test, y_test)


# Run Metrics Visualisation for the four models chosen.
vs.visualize_classification_performance(results)

# Feature Importance

## Figuring out which features provide the most predictive power.

### Our intention here is to find out the relationships between only a few crucial features and the target label to simplify our understanding of the phenomenon.
### Thus, all we want to do is identify a small number of features that most strongly predit the quality of wines.

# Import a supervised learning model that has 'feature_importances_'
model = RandomForestClassifier(max_depth = None, random_state = None)

# Train the supervised model on the training set using .fit(X_train, y_train)
model = model.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_
importances = model.feature_importances_

print(X_train.columns)
print(importances)

# Visualisation
vs.feature_plot(importances, X_train, y_train)

# Hyperparameter tuning using GridSearchCV

# initlialise the classifier
clf = RandomForestClassifier(max_depth = None, random_state = None)

'''
n_estimators
    Number of Trees in the Forest
max_features
    The number of features to consider when looking for the best split
max_depth
    The maximum depth of the tree
'''
parameters = {'n_estimators': [10,20,30], 'max_features': [3,4,5, None], 'max_depth': [5,6,7, None]}

# F-Beta score
scorer = make_scorer(fbeta_score, beta = 0.5, average = "micro")

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring = scorer)

# Fill the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimised model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before and after scores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average="micro")))
print("\nOptimized Model\n------")
print(best_clf)
print("\nFinal accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5,  average="micro")))


## Final Model testing
"""Give inputs in this order: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
total sulfur dioxide, density, pH, sulphates, alcohol

"""
wine_data = [[8, 0.2, 0.16, 1.8, 0.065, 3, 16, 0.9962, 3.42, 0.92, 9.5],
            [8, 0, 0.16, 1.8, 0.065, 3, 16, 0.9962, 3.42, 0.92, 1 ],
            [7.4, 2, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 0.6]]
               
# Show predictions
for i, quality in enumerate(best_clf.predict(wine_data)):
    print("Predicted quality for Wine {} is: {}".format(i+1, quality))