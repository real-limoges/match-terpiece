import sys
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATA_DIR = '../data/'
IMAGE_DIR = '../images/'


def build_tree(df, metric):
    '''
    INPUTS: Pandas DataFrame, Choice of Metric Space String
    OUTPUTS: Returns the built AnnoyIndex tree, returns a dictionary
             mapping index numbers to the DataFrame's index

    Builds a ANN tree using Spotify's ANNoy library. Metric is the
    metric space (either euclidean or angular)
    '''
    tree = AnnoyIndex(len(df.iloc[0,:].values), metric=metric)

    indexes = {}

    for i in xrange(len(df)):
        v = df.iloc[i,:]
        indexes[i] = v.name
        tree.add_item(i, v.values)

    tree.build(10)
    return (tree, indexes)

def show_neighbors(tree, indexes, index, k=5):
    '''
    INPUTS: Built AnnoyIndex, Dictionary of Ints -> Strings, 
            Numpy Array of size (4096,)
    OUTPUTS: Side Effects Only (Shows Current Image and that of the
             k closest neighbors)
    '''
    
    nns = tree.get_nns_by_vector(index, k+1)
    
    for i in nns:
        img = mpimg.imread(IMAGE_DIR + indexes[i])
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
    else:
        raise Exception("Please provide a dataset")
    
    tree_e, indexes_e = build_tree(df, "euclidean")
    tree_a, indexes_a = build_tree(df, "angular")

    lookup=0

    while lookup != -1:
        lookup = int(raw_input('Enter an image number: '))
        if lookup == -1:
            pass
        i = df.iloc[lookup,:].values

        show_neighbors(tree_e, indexes_e, i)
