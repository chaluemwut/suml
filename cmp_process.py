from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

from dataset_loader import DataSetLoader
from sklearn import cross_validation

class CompareProcess(object):
    data_size = [0.75, 0.50, 0.25]
    reperating_loop = 500
    
    def __init__(self):
        pass
    
    def gen_ml_lst(self):
        base_estimation = [2, 32, 64, 1024]
        nb = []
        nb.append(GaussianNB())
        
        rf = []
        for i in base_estimation:
            rf.append(RandomForestClassifier(n_estimators=i))
        
        return {'nb':nb,
                'rf':rf}
        
    def load_dataset(self):
        loader = DataSetLoader()
        lst = loader.loadData()
        return lst
    
    def cross_validation(self, ml_lst, x, y):
        pass
    
    def process(self):
        max_model = []
        ml_lst = self.gen_ml_lst()
        dataset_lst = self.load_dataset()
        for ml_key, ml_value in ml_lst.iteritems():
            for data_key, data_value in dataset_lst.iteritems():
                for d_size in self.data_size:
                    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.33, random_state=42)
    