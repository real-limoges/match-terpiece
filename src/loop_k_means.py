import os
import pandas as pd
import sys
import cPickle as pickle
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import StandardScaler

DATA_DIR = '../data/'
RESULTS_DIR = '../results'

def find_best(df, cls, norm):
    '''
    INPUTS: Pandas DataFrame, String of which dataset is being used, 
            Boolean if data is normalized
    OUTPUTS: Prints score to screen. Saves model to RESULTS_DIR
    '''
    if norm == True:
        df = StandardScaler(copy=False).fit_transform(df)
    
    files = [f for f in os.listdir(RESULTS_DIR)
                   if f.endswith('{}_{}.pkl'.format(cls,norm))]
    print files
    scores = []
    '''
    for f in files:
        model = pickle.load(open(f, 'rb'))
        labels = model.predict(df)
        score = silhouette_score(df.values, labels, sample_size=10000)

        name = f[:2].rstrip('_')
        scores.append((score, name))
        print "{} {} {} {}".format(f.split('_')[0], float(score), cls, norm)
        
        del labels
        del model
    '''
if __name__ == '__main__':
    if len(sys.argv) == 2:
        df = pd.read_csv(DATA_DIR + sys.argv[1], index_col=0)
        cls = sys.argv[1][:3]
    else:
        raise Exception("Please enter valid files")
    
    find_best(df, cls, 'False')
    find_best(df, cls, 'True')

