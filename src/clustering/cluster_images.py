import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, MeanShift
import cPickle as pickle
from sklearn import metrics
import sys


# Sklearn uses a depreciated connection to Fortran - raises a warning
import warnings
warnings.filterwarnings('ignore')

directory = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(directory, '../../data/')
IMAGE_DIR = os.path.join(directory, '../../images/')
RESULTS_DIR = os.path.join(directory, '../../results/')


def run_k_means(df):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Fitted K-Means model that has the highest 
             Silhouette Coefficient

    Iterates through a series of k hyperparameters (5 to 20) and computes
    the Silhouette Coefficient for each (a measure of how dense the clusters
    are). Returns the model that has the densest clusters
    '''

    models, scores = [], []

    for k in xrange(5, 20):
        print "Clustering with {} as the hyperparameter".format(k)
        model = model_k_means(df, k)
        models.append(model)
        labels = model.labels_
        scores.append(metrics.silhouette_score(df, labels,
                                               metric='euclidean'))

    index = np.argmax(np.array(scores))

    return models[index]


def model_k_means(df, k):
    '''
    INPUTS: Pandas Dataframe, Number of Clusters (Int)
    OUTPUTS: Fitted K-Means Model
    '''
    clstr = KMeans(n_clusters=k, init='k-means++', random_state=42)
    return clstr.fit(df)


def run_mean_shift(df):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Returns a fitted MeanShift object
    '''
    model = MeanShift(min_bin_freq=10, cluster_all=False, n_jobs=-1)
    return model.fit(df)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        cls = sys.argv[1][:3]
    else:
        raise Exception('Please provide a dataset')

    # Builds K-Means
    k_means = run_k_means(df)
    with open(RESULTS_DIR + 'k_means_model_{}.pkl'.format(cls), 'wb') as f:
        pickle.dump(k_means, f)

    # Builds MeanShift
    mean_shift = run_mean_shift(df)
    with open(RESULTS_DIR + 'mean_shift_model_{}.pkl'.format(cls),
              'wb') as f:
        pickle.dump(mean_shift, f)
