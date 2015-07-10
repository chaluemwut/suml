from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from dataset_loader import DataSetLoader
from sklearn import cross_validation
import copy, random, time
import numpy as np

ml = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree', 'svm']

class CompareProcess(object):
    data_size = [0.75, 0.50, 0.25]
    reperating_loop = 10
    
    def __init__(self):
        pass
    
    def gen_ml_lst(self):
        base_estimation = [2,4]#[2, 32, 64, 1024]
        knn = [2, 32, 64, 1024]
        
        random_lst = []
        boosted_lst = []
        bagging_lst = []
        for i in base_estimation:
            random_lst.append(RandomForestClassifier(n_estimators=i))
            boosted_lst.append(GradientBoostingClassifier(n_estimators=i))
            bagging_lst.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=i))
        
        knn_lst = []
        for i in knn:
            knn_lst.append(KNeighborsClassifier(n_neighbors=i))
            
        svm_lst = []
        svm_lst.append(SVC(kernel='linear'))
        svm_lst.append(SVC(kernel='poly', degree=2))
        svm_lst.append(SVC(kernel='poly', degree=3))
        svm_lst.append(SVC(kernel='sigmoid'))
        
        return {ml[0]:bagging_lst,
                ml[1]:boosted_lst,
                ml[2]:random_lst,
                ml[3]:[GaussianNB()],
                ml[4]:knn_lst,
                ml[5]:[DecisionTreeClassifier()],
                ml[6]:svm_lst
        }
        
    def load_dataset(self):
        loader = DataSetLoader()
        lst = loader.loadData()
        return lst
    
    def cross_validation(self, ml_lst, x, y):
        score_lst = []    
        for ml in ml_lst:
            scores = cross_validation.cross_val_score(ml, x, y, cv=5)
            score_lst.append(scores.mean())
        np_score = np.array(score_lst)
        max_idx = np_score.argmax()
        return ml_lst[max_idx]
    
    def process(self):
        ml_lst = self.gen_ml_lst()
        dataset_lst = self.load_dataset()
        for ml_key, ml_value in ml_lst.iteritems():
            for data_key, data_value in dataset_lst.iteritems():
                x_data = data_value[0]
                y_data = data_value[1]
                for d_size in self.data_size:
                    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=d_size, random_state=42)
                    ml = self.cross_validation(ml_value, x_train, y_train)
                    acc_lst = []
                    f1_lst = []
                    time_pred = []
                    total_ins = []
                    for i in range(0, self.reperating_loop):
                        ran_num = random.randint(1,10000)
                        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=d_size, random_state=ran_num)
                        ml_c = copy.deepcopy(ml)
                        ml_c.fit(x_train, y_train)
                        start = time.time()
                        y_pred = ml_c.predict(x_test)
                        total_time = time.time()-start
                        acc = accuracy_score(y_test, y_pred)
                        fsc = f1_score(y_test, y_pred)
                        acc_lst.append(acc)
                        f1_lst.append(fsc)
                        time_pred.append(total_time)
                        total_ins.append(len(y_test))
                    print 'data size ', d_size, ' data set ', data_key, ' acc ', acc_lst

def mainCmp():
    obj = CompareProcess()
    obj.process()
    
if __name__ == '__main__':
    mainCmp()    
