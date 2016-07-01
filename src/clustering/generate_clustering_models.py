import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
from sklearn import metrics
import sys

# Sklearn uses a depreciated connection to Fortran - raises a warning
import warnings
warnings.filterwarnings('ignore')

directory = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(directory, '../../data/')
IMAGE_DIR = os.path.join(directory, '../../images/')


def run_k_means(df, cls, norm=False):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Fitted K-Means model that has the highest Silhouette
    Coefficient

    Iterates through a series of k hyperparameters (5 to 20) and computes
    the Silhouette Coefficient for each (a measure of how dense the clusters
    are). Returns the model that has the densest clusters
    '''
    if norm == True:
        df = StandardScaler(copy=False).fit_transform(df)
    scores = []
    for k in xrange(5, 20):
        print "Clustering with {} as the hyperparameter".format(k)
        model = model_k_means(df, k)

        with open(RESULTS_DIR + 'k_{}_{}_{}.pkl'.format(k, cls, norm),
                  'wb') as f:
            pickle.dump(model, f)


def model_k_means(df, k):
    '''
    INPUTS: Pandas Dataframe, Number of Clusters (Int)
    OUTPUTS: Fitted K-Means Model
    '''
    clstr = MiniBatchKMeans(n_clusters=k, random_state=42,
                            batch_size=100, verbose=0, compute_labels=False,
                            init='random')
    return clstr.fit(df)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        cls = sys.argv[1][:3]
    else:
        raise Exception("Please provide a data file")

    run_k_means(df, cls, norm=False)
    run_k_means(df, cls, norm=True)
