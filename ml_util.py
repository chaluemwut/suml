'''
Created on Jul 13, 2015

@author: off
'''
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import copy

class MLUtil(object):
    
    @staticmethod
    def cross_validation(ml, x, y, cv=5):
        result = []
        kf = KFold(len(x), n_folds=cv)
        for train, test in kf:
            ml_c = copy.deepcopy(ml)
            ml_c.fit(x[train], y[train])
            y_true = y[test]
            y_pred = ml_c.predict(x[test])
            result.append(accuracy_score(y_true, y_pred))
        return result