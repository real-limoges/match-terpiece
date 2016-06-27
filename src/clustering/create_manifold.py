from sklearn.preprocessing import StandardScaler, TruncatedSVD

from sklearn import (manifold, datasets, decomposition,ensemble,
                    discriminant_analysis, random_projection)

import sys
import numpy as np


RANDOM_STATE=42


def plot_manifold(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)



def run_TSNE(df):
    tsne = manifold.TSNE(n_components=2, method='exact',
                         random_state=RANDOM_STATE)

    X_tsne = tsne.fit_transform(df) 

def run_Isomap(df):
    isomap = manifold.Isomap(n_neighbors=30, n_components=2)

    X_iso = isomap.fit_transform(df)

def preprocess_data(df):
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

