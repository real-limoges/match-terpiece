import os
import pandas as pd
import sys
import cPickle as pickle
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = '../results'

def find_best(df, cls, norm):

    if norm == True:
        df = StandardScaler(copy=False).fit_transform(df)
    
    files = [f for f in os.listdir('.')
                    if cls in f and norm in f]

    scores = []

    for f in files:
        model = pickle.load(open(f, 'rb'))
        labels = model.predict(df)
        score = silhouette_score(df.values, labels, sample_size=10000)

        name = f[:2].rstrip('_')
        scores.append((score, name))
        print "{} {} {} {}".format(f.split('_')[0], float(score), cls, norm)
        
        del labels
        del model

if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(sys.argv[1], index_col=0)
        cls = sys.argv[1][:3]
    else:
        raise Exception("Please enter valid files")
    
    find_best(df, cls, 'False')
    find_best(df, cls, 'True')

