import os
import cPickle as pickle
from PIL import Image as II
from math import sqrt
import pandas as pd


IMAGE_DIR = '../../tmp/images/'
DATA_DIR = '../data/'
RESULTS_DIR = '../results/'

def is_square(num):
    root = sqrt(num)
    if int(root + 0.5)**2 == num:
        return True
    return False

def make_file_name(image_name):
    return os.getcwd() + '/' + IMAGE_DIR + '/' + image_name + '.jpg'


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
    background = II.new('RGB', (224 * num, 224 * num), (255, 255, 255))

    for i in xrange(num):
        for j in xrange(num):
            img = II.open(images[i][j])
            background.paste(img, (i * 224, j * 224))
    background.save(RESULTS_DIR + cluster_name + '.png')


if __name__ = '__main__':
    df = pd.read_csv(DATA_DIR + 'clusters.csv')

    with open(RESULTS_DIR + 'k_means_model.pkl', 'rb') as f:
        clstr = pickle.load(f)

    fitted_df = clstr.predict(df)
    diff_df = clstr.transform(df)
    '''
    for cluster in range(clstr.n_clusters):
        clst = np.where(fitted_df == cluster)
        images = np.array([make_file_name(map_files[index][1])
                           for index in clst[0][:25]])

        plot_images(images, "Cluster {}".format(cluster))
    '''