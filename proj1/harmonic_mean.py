import numpy as np
from scipy.stats import hmean

class HarmonicMeanClassifier(object):

    def __init__(self,list_classifiers):
        self.estimators_ = list_classifiers

    def fit(self,X,y):
        for clf in self.estimators_:
            clf.fit(X, y)

    def predict_proba(self,X):
        res = np.zeros((X.shape[0],len(self.estimators_)))
        for i,clf in enumerate(self.estimators_):
            res[:,i] = clf.predict_proba(X)[:,1]
        res = hmean(res, axis=1)
        res = np.stack((res, 1-res), axis=1)
        return res