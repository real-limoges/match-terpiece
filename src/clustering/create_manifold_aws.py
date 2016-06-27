from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomTreesEmbedding
import pandas as pd
from sklearn import manifold
import numpy as np

sample = pd.read_csv('../data/fc1.csv', index_col=0)

print "*************** HESSIAN **************"
for i in xrange(2, 10):
    hessian = manifold.LocallyLinearEmbedding(n_neighbors=55, 
            n_components=i, random_state=42, method='hessian',
           eigen_solver='dense')
    X_hess = hessian.fit_transform(sample)
    print i, " ", hessian.reconstruction_error_

print "************ STANDARD ****************"
for i in xrange(2, 10):
    standard = manifold.LocallyLinearEmbedding(n_neighbors=55, 
            n_components=i, random_state=42, method='standard',
            eigen_solver='dense')
    X_hess = standard.fit_transform(sample)
    print i, " ", standard.reconstruction_error_


print "************ LTSA  ****************"
for i in xrange(2, 10):
    ltsa = manifold.LocallyLinearEmbedding(n_neighbors=55, 
            n_components=i, random_state=42, method='ltsa',
            eigen_solver='dense')
    X_hess = ltsa.fit_transform(sample)
    print i, " ", ltsa.reconstruction_error_


for i in xrange(2, 10):
    mds = manifold.MDS(n_components=i, random_state=42, n_jobs=-1)
    X_mds = mds.fit_transform(sample)
    print i, ' ', mds.stress_



