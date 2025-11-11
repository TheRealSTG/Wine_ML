###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
# Only run IPython magic if we're in an IPython environment
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
###########################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, feature_label, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    sns.set()
    sns.set_style("whitegrid")
    # Create figure
    fig = plt.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate([feature_label]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Total Number")
        ax.set_ylim((0, 1500))
        ax.set_yticks([0, 200, 400, 600, 800, 1000])
        ax.set_yticklabels([0, 200, 400, 600, 800, ">1000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def visualize_classification_performance(results):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - results: a list of dictionaries of the statistic results from 'train_predict_evaluate()'
    """
  
    # Create figure with better sizing
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    
    # Constants - improved colors and bar width
    bar_width = 0.18
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code with edge color for better visibility
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], 
                                 width=bar_width, color=colors[k], 
                                 edgecolor='black', linewidth=0.7, alpha=0.85)
                ax[j//3, j%3].set_xticks([0.27, 1.27, 2.27])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"], fontsize=11, fontweight='bold')
                ax[j//3, j%3].set_xlabel("Training Set Size", fontsize=12, fontweight='bold')
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
                ax[j//3, j%3].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add unique y-labels with better formatting
    ax[0, 0].set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax[0, 1].set_ylabel("Accuracy Score", fontsize=12, fontweight='bold')
    ax[0, 2].set_ylabel("F-score", fontsize=12, fontweight='bold')
    ax[1, 0].set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax[1, 1].set_ylabel("Accuracy Score", fontsize=12, fontweight='bold')
    ax[1, 2].set_ylabel("F-score", fontsize=12, fontweight='bold')
    
    # Add titles with better styling
    ax[0, 0].set_title("Model Training Time", fontsize=13, fontweight='bold', pad=10)
    ax[0, 1].set_title("Training Accuracy", fontsize=13, fontweight='bold', pad=10)
    ax[0, 2].set_title("Training F-score", fontsize=13, fontweight='bold', pad=10)
    ax[1, 0].set_title("Model Prediction Time", fontsize=13, fontweight='bold', pad=10)
    ax[1, 1].set_title("Testing Accuracy", fontsize=13, fontweight='bold', pad=10)
    ax[1, 2].set_title("Testing F-score", fontsize=13, fontweight='bold', pad=10)
    
    # Add horizontal reference lines
    ax[0, 1].axhline(y=0.5, xmin=-0.1, xmax=3.0, linewidth=1.5, color='gray', linestyle='--', alpha=0.5)
    ax[1, 1].axhline(y=0.5, xmin=-0.1, xmax=3.0, linewidth=1.5, color='gray', linestyle='--', alpha=0.5)
    ax[0, 2].axhline(y=0.5, xmin=-0.1, xmax=3.0, linewidth=1.5, color='gray', linestyle='--', alpha=0.5)
    ax[1, 2].axhline(y=0.5, xmin=-0.1, xmax=3.0, linewidth=1.5, color='gray', linestyle='--', alpha=0.5)
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1.05))
    ax[0, 2].set_ylim((0, 1.05))
    ax[1, 1].set_ylim((0, 1.05))
    ax[1, 2].set_ylim((0, 1.05))

    # Create patches for the legend with better positioning
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner, edgecolor='black', linewidth=0.5))
    plt.legend(handles=patches, bbox_to_anchor=(0.5, 2.65), 
               loc='upper center', borderaxespad=0., ncol=4, 
               fontsize=12, frameon=True, shadow=True, fancybox=True)
    
    # Aesthetics with updated title
    plt.suptitle("Performance Metrics for Four Supervised Learning Models", 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(pad=2, w_pad=3, h_pad=4.0)
    plt.show()
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:11]]
    values = importances[indices][:11]

    sns.set()
    sns.set_style("whitegrid")

    # Creat the plot
    fig = plt.figure(figsize = (12,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(11), values, width = 0.2, align="center", label = "Feature Weight")
    # plt.bar(np.arange(11) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
    #       label = "Cumulative Feature Weight")
    plt.xticks(np.arange(11), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  


