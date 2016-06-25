import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans
import cPickle as pickle
from sklearn import metrics
import sys

#Sklearn uses a depreciated connection to Fortran - raises a warning
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data/'
RESULTS_DIR = '../results/'


def run_k_means(df, cls):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Fitted K-Means model that has the highest Silhouette 
    Coefficient

    Iterates through a series of k hyperparameters (5 to 20) and computes
    the Silhouette Coefficient for each (a measure of how dense the clusters
    are). Returns the model that has the densest clusters
    '''

    scores = []
    for k in xrange(5, 20):
        print "Clustering with {} as the hyperparameter".format(k)
        model = model_k_means(df, k)

        with open(RESULTS_DIR + '{}_means_model_{}.pkl'.format(k,cls), 
                    'wb') as f:
            pickle.dump(model, f)

def model_k_means(df, k):
    '''
    INPUTS: Pandas Dataframe, Number of Clusters (Int)
    OUTPUTS: Fitted K-Means Model
    '''
    clstr = MiniBatchKMeans(n_clusters=k, random_state=42,
            batch_size=50, verbose=1, compute_labels=False)
    return clstr.fit(df)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        cls = sys.argv[1][:3]
    else:
        raise Exception("Please provide a data file")
    
    k_means = run_k_means(df, cls) 
