import sys
from sklearn.decomposition import PCA, FactorAnalysis, NMF
import numpy as np
import pandas as pd

DATA_DIR = '../data/'

def compute_PCA(df):
    pca = PCA()
    pca.fit(df)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = len(cumsum[cumsum < 0.9])
    return PCA(n_components = n_comp).fit_transform(df)

def compute_FA(df):
    FA = FactorAnalysis()
    return FA.fit_transform(df)

def compute_NMF(df):
    nmf = NMF(n_components=300)
    return nmf.fit_transform(df)


def reduce_dimensions(df, cls):
    pca_df = pd.DataFrame(compute_PCA(df))
    pca_df.set_index(df.index, inplace=True)
    pca_df.to_csv(DATA_DIR + 'pca_{}_clusters.csv'.format(cls), index_col=0)

    FA_df = pd.DataFrame(compute_FA(df))
    FA_df.set_index(df.index, inplace=True)
    FA_df.to_csv(DATA_DIR + 'fa_{}_clusters.csv'.format(cls), index_col=0)

    nmf_df = pd.DataFrame(compute_NMF(df))
    nmf_df.set_index(df.index, inplace=True)
    nmf_df.to_csv(DATA_DIR + 'nmf_{}_clusters.csv'.format(cls), index_col=0)

if __name__ == '__main__':
    for i in sys.argv:
        if i != __file__:
            reduce_dimensions(df, i[:3])
    
