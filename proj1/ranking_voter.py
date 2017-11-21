import numpy as np

class RankVotingClassifier(object):

    def __init__(self,list_classifiers):
        self.estimators_ = list_classifiers
        self.classes_ = [0,1]

    def fit(self,X,y):

        for clf in self.estimators_:
            clf.fit(X,y)

    def predict_proba(self,X):

        res = np.zeros((X.shape[0],len(self.estimators_)))
        for i,clf in enumerate(self.estimators_):
            res[:,i] = np.argsort(clf.predict_proba(X)[:,1])

        p1 = np.median(res,axis=1)/np.shape(X)[0]
        p0 = 1-p1
        return np.concatenate((p0,p1),axis=1)