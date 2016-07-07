import sys
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
#import matplotlib.image as mpimg
import os
import cPickle as pickle
from sklearn.preprocessing import StandardScaler


directory = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(directory, '../../data/')
IMAGE_DIR = os.path.join(directory, '../../images/')


def build_tree(df, metric):
    '''
    INPUTS: Pandas DataFrame, Choice of Metric Space String
    OUTPUTS: Returns the built AnnoyIndex tree, returns a dictionary
             mapping index numbers to the DataFrame's index

    Builds a ANN tree using Spotify's ANNoy library. Metric is the
    metric space (either euclidean or angular)
    '''
    tree = AnnoyIndex(len(df.iloc[0, :].values), metric=metric)

    indexes = {}

    for i in xrange(len(df)):
        v = df.iloc[i, :]
        indexes[i] = v.name
        tree.add_item(i, v.values)

    tree.build(50)

    tree.save(DATA_DIR + 'tree_' + metric + '.ann')
    with open(DATA_DIR + 'indexes_' + metric, 'wb') as f:
        pickle.dump(indexes, f)

    return (tree, indexes)


def show_neighbors(tree, indexes, index, k=5):
    '''
    INPUTS: Built AnnoyIndex, Dictionary of Ints -> Strings,
            Numpy Array of size (4096,)
    OUTPUTS: Side Effects Only (Shows Current Image and that of the
             k closest neighbors)
    '''

    nns = tree.get_nns_by_vector(index, k + 1)

    for i in nns:
        img = mpimg.imread(IMAGE_DIR + indexes[i])
        plt.imshow(img)
        plt.show()


def get_tree_index(metric='angular', size=4096):
    '''
    INPUT: Optional parameters for the metric space and size of AnnoyIndex 
    OUTPUT: AnnoyIndex tree, dictionary of node assignment to image names
    '''
    tree = AnnoyIndex(size, metric=metric)
    tree.load(DATA_DIR + 'tree_' + metric + '.ann')

    with open(DATA_DIR + 'indexes_' + metric, 'rb') as f:
        indexes = pickle.load(f)

    return tree, indexes

if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        scaler = StandardScaler().fit(df)
        df = pd.DataFrame(scaler.transform(df), index=df.index,
                          columns=df.columns)
    else:
        raise Exception("Please provide a dataset")

    tree_a, indexes_a = build_tree(df, "angular")
