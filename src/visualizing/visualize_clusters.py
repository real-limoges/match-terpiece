import os
import cPickle as pickle
from PIL import Image as II
from math import sqrt
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = '../images/'
DATA_DIR = '../data/'
RESULTS_DIR = '../results/'
MAX_NUM = 49


def is_square(num):
    root = sqrt(num)
    if int(root + 0.5)**2 == num:
        return True
    return False

def make_file_name(image_name):
    return  IMAGE_DIR  + image_name


def plot_images(images, cls, norm, k, cluster): 
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
    background = II.new('RGB', (224 * num, 224 * num), (255, 255, 255))

    for i in xrange(num):
        for j in xrange(num):
            img = II.open(images[i][j])
            background.paste(img, (i * 224, j * 224))

    path = "{}{}/{}/{}/{}.png".format(RESULTS_DIR, cls, norm, k, cluster)
    print path
    background.save(path)

def run_plot(df, cls, norm, model=False):
    if model == False:
        find_k = pickle.load(open("{}{}_{}.pkl".format(RESULTS_DIR, 
                                cls, norm)))
        model = pickle.load(open("{}k_{}_{}_{}.pkl".format(RESULTS_DIR, 
                                find_k[0][0], cls, norm), 'rb'))
    if norm == True:
        df1 = StandardScaler(copy=False).fit_transform(df)
        df = pd.DataFrame(df1, index=df.index, columns=df.columns)
        del df1
    
    fitted_df = model.predict(df)

    for cluster in range(model.n_clusters):
        clst = np.where(fitted_df == cluster)[0]
        image_names = list(df.iloc[clst[:MAX_NUM], :].index)
        images = np.array([make_file_name(image) for image in image_names])
        plot_images(images, cls, norm, model.n_clusters, cluster) 


if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        cls = sys.argv[1][:3]
        run_plot(df, cls, False)
        run_plot(df, cls, True)
    elif len(sys.argv) == 3:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        cls = sys.argv[1][:3]
        model = pickle.load(open(RESULTS_DIR + sys.argv[2]))
        if sys.argv[2].split('_')[3].startswith('T'):
            run_plot(df, cls, True, model)
        else:
            run_plot(df, cls, False, model)
    else:
        raise Exception("Incorrect number of parameters specified")
