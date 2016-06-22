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


if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR + 'clusters.csv', index_col=0)
    
    '''
    pca_df = pd.DataFrame(compute_PCA(df))
    pca_df.set_index(df.index, inplace=True)
    pca_df.to_csv(DATA_DIR + 'pca_clusters.csv', index=True)

    FA_df = pd.DataFrame(compute_FA(df))
    FA_df.set_index(df.index, inplace=True)
    FA_df.to_csv(DATA_DIR + 'fa_clusters.csv', index=True)
    '''

    nmf_df = pd.DataFrame(compute_NMF(df))
    nmf_df.set_index(df.index, inplace=True)
    nmf_df.to_csv(DATA_DIR + 'nmf_clusters.csv', index=True)
