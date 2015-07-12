from multiprocessing import Pool, Process
from config import Config

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn import linear_model
from sklearn import cross_validation
from dataset_loader import DataSetLoader

class ParallelProcess(object):
 
    def gen_ml_lst(self):
        random_lst = []
        boosted_lst = []
        bagging_lst = []
        for i in Config.base_estimation:
            random_lst.append(RandomForestClassifier(n_estimators=i))
            boosted_lst.append(GradientBoostingClassifier(n_estimators=i))
            bagging_lst.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=i))
        knn_lst = []
            
        return {
                    Config.ml_name[0]:bagging_lst,
                    Config.ml_name[1]:boosted_lst,
                    Config.ml_name[2]:random_lst,
                    Config.ml_name[3]:[GaussianNB()],
                    Config.ml_name[4]:knn_lst,
                    Config.ml_name[5]:[DecisionTreeClassifier()],
                    Config.ml_name[6]:[linear_model.SGDClassifier()]
        }
            
    def process_by_ml_name(self, ml):
        print 'start ', ml
        loader = DataSetLoader()
        x, y = loader.loadData()[DataSetLoader.dataset_name[0]]
        score_lst = []    
        for ml in ml:
            print 'start cross val'
            scores = cross_validation.cross_val_score(ml, x, y, cv=5)
            print 'end cross val'
            score_lst.append(scores.mean())        
        print 'end '
        
    def process(self):
        ml_lst = self.gen_ml_lst()
        pool = Pool(processes=2)
        for m in Config.ml_name:
            ml = ml_lst[m]
            pool.map(self.process_by_ml_name,(ml,))
            pool.close()
            pool.join()
#             p = Process(target=self.process_by_ml_name, args=(ml,))
#             p.start()
#             p.join()
    
if __name__ == '__main__':
    obj = ParallelProcess()
    obj.process()
