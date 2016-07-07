from sklearn.preprocessing import StandardScaler, TruncatedSVD

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import sys
import numpy as np
import os

RANDOM_STATE = 42


def run_TSNE(df):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Fitted t-SNE dataset (numpy array)
    '''

    tsne = manifold.TSNE(n_components=2, method='exact',
                         random_state=RANDOM_STATE)

    X_tsne = tsne.fit_transform(df)


def run_Isomap(df):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Fitted Isomap dataset (numpy array)
    '''
    isomap = manifold.Isomap(n_neighbors=30, n_components=2)

    X_iso = isomap.fit_transform(df)


def preprocess_data(df):
    '''
    INPUTS: Pandas Dataframe
    OUTPUTS: Pandas Dataframe

    Function takes a dataframe, normalizes each column, and then takes
    the first 50 components from sklearn's TruncatedSVD.
    '''

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(scaled_data, index=df.index,
                               columns=df.columns)

    sample = scaled_data.sample(n=5000, random_state=RANDOM_STATE)
    svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE,
                       tol=0.0)

    return pd.DataFrame(svd.fit_transform(sample), index=sample.index)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        sample = df.sample(n=5000, random_state=RANDOM_STATE)
    else:
        raise Exception("Please enter a vaild file")

    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df),
                      index=df.index, columns=df.columns)

    sample = preprocess_data(df)
    del df
