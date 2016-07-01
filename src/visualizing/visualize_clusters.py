import os
import cPickle as pickle
from PIL import Image as II
from math import sqrt
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

directory = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(directory, '../../data/')
IMAGE_DIR = os.path.join(directory, '../../images/')
RESULTS_DIR = os.path.join(directory, '../../results/')

MAX_NUM = 49


def is_square(num):
    '''
    INPUTS: Number (integer)
    OUTPUTS: Boolean if number is perfect square
    '''
    
    root = sqrt(num)
    if int(root + 0.5)**2 == num:
        return True
    return False


def make_file_name(image_name):
    '''
    INPUTS: Name of image
    OUTPUTS: String of the path to the image
    '''
    
    return IMAGE_DIR + image_name


def plot_images(images, cls, norm, k, cluster):
    '''
    INPUT: Numpy Array (Image Names), Cluster Name (String), 
           Boolean (Whether data is normalized), K (Nubmer of Clusters),
           Num (Cluster Name)
    OUTPUT: Side Effects Only (Saves PNG of top N images for a cluster)

    Takes in a list of image names. Converts it into a perfect square
    and displays that perfect square of images in a PNG file.
    '''

    if is_square(len(images)) is not True:
        # Removes one image at a time until a perfect square is reached
        converged = False
        num = len(images)
        
        while converged == False:
            num -= 1
            converged = is_square(num)
        
        # Reshapes images into (num, num) numpy array 
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
    '''
    INPUTS: Pandas DataFrame, String Representing Layer, Boolean whether
            the data is normalized, Pickle Model to open (optional)
    OUTPUTS: Side Effects Only (Calls 

    Opens model and data. Shows the cluster centers by printing them to
    file
    '''
    
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
