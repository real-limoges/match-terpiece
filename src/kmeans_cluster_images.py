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

IMAGE_DIR = 'images/'
DATA_DIR = 'data/'
RESULTS_DIR = 'results/'

def is_square(num):
    root = sqrt(num)
    if int(root + 0.5)**2 == num:
        return True
    return False
    
def make_file_name(image_name):
    return os.getcwd() + '/' +  IMAGE_DIR + '/' + image_name + '.jpg'

def plot_images(images, cluster_name):
    '''
    INPUT: Numpy Array (Image Names), Cluster Name (String), Num (int)
    OUTPUT: Side Effects Only (Saves PNG of top N images for a cluster)

    Takes in a list that is 
    '''
    
    if is_square(len(images)) is not True:
        converged = False
        num = len(images)
        while converged == False:
            num -= 1
            converged = is_square(num)
        images = images[:num]
        num = int(sqrt(len(images)))
        images = images.reshape(num, num)
    else:
        num = int(sqrt(len(images)))
        images = images.reshape(num, num)
    h, w, = 224, 224
    background = II.new('RGB', (224*num, 224*num), (255, 255, 255))
    
    for i in xrange(num):
        for j in xrange(num):
            img = II.open(images[i][j])
            background.paste(img, (i*224, j*224))
    background.save(RESULTS_DIR + cluster_name + '.png')

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
        models.append(define_cluster(df, k))
    
    for model in models:
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
    df = pd.read_csv('data/cluster.csv')
    file_names = [image[:-4] for image in os.listdir(IMAGE_DIR) 
            if image.endswith('.jpg')]
    map_files = zip(xrange(1000), file_names) 
    #k_means = find_max_cluster(df)
    #with open('k_means_model.pkl', 'wb') as f:
    #    pickle.dump(k_means, f)
    
    with open('k_means_model.pkl', 'r') as f:
        clstr = pickle.load(f)

    fitted_df = clstr.predict(df)
    dist_df = clstr.transform(df)
    
    for cluster in range(clstr.n_clusters):
        clst = np.where(fitted_df == cluster)
        images = np.array([make_file_name(map_files[index][1])
                          for index in clst[0][:25]])

        plot_images(images, "Cluster {}".format(cluster)) 
