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


def process_by_ml_name(ml):
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

def process():
    random_lst = []
    boosted_lst = []
    bagging_lst = []
    for i in Config.base_estimation:
        random_lst.append(RandomForestClassifier(n_estimators=i))
        boosted_lst.append(GradientBoostingClassifier(n_estimators=i))
        bagging_lst.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=i))
    knn_lst = []
            
    ml_map = {
                    Config.ml_name[0]:bagging_lst,
                    Config.ml_name[1]:boosted_lst,
                    Config.ml_name[2]:random_lst,
                    Config.ml_name[3]:[GaussianNB()],
                    Config.ml_name[4]:knn_lst,
                    Config.ml_name[5]:[DecisionTreeClassifier()],
                    Config.ml_name[6]:[linear_model.SGDClassifier()]
        }
    
    pool = Pool(processes=10)
    for m in Config.ml_name:
        ml = ml_map[m]
        pool.map(process_by_ml_name, (ml,))
        pool.close()
        pool.join()
    
if __name__ == '__main__':
    process()

# http://aaren.me/notes/2012/04/embarassingly_parallel_python