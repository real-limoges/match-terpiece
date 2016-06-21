import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from PIL import Image as II
from math import sqrt
from sklearn import metrics
import cPickle as pickle

IMAGE_DIR = '../../tmp/images/'
DATA_DIR = '../data/'
RESULTS_DIR = '../results/'


def find_max_cluster(df):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Fitted K-Means model that has the highest Silhouette Coefficient

    Iterates through a series of k hyperparameters (5 to 20) and computes
    the Silhouette Coefficient for each (a measure of how dense the clusters
    are). Returns the model that has the densest clusters
    '''

    models, scores = [], []

    for k in xrange(5, 20):
        print "Clustering with {} as the hyperparameter".format(k)
        model = define_cluster(df, k)
        models.append(model)
        labels = model.labels_
        scores.append(metrics.silhouette_score(df, labels, metric='euclidean'))
    
    index = np.argmax(np.array(scores))
    
    return models[index]


def define_cluster(df, k):
    '''
    INPUTS: Pandas Dataframe, Number of Clusters (Int)
    OUTPUTS: Fitted K-Means Model
    '''
    clstr = KMeans(n_clusters=k, init='k-means++', random_state=42)
    return clstr.fit(df)

if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR + 'clusters.csv')
    file_names = [image[:-4] for image in os.listdir(IMAGE_DIR)
                  if image.endswith('.jpg')]

    df.set_index(pd.Series(file_names), inplace=True)
    # map_files = zip(xrange(1000), file_names)
    k_means = find_max_cluster(df)
    
    with open(RESULTS_DIR + 'k_means_model.pkl', 'wb') as f:
        pickle.dump(k_means, f)
