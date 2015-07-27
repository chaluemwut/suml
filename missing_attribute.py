from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn import linear_model
from config import Config

from dataset_loader import DataSetLoader
from sklearn import cross_validation
import copy, random, time, logging, sys, pickle
import numpy as np
from tabulate import tabulate
from svm import LibSVMWrapper
from log_file import LogFile

class MissingAttribute(object):
    data_size = [0.75, 0.50, 0.25]
    dataset_name_lst = ['adult', 'heart', 'letter', 'austra', 'german', 'sat', 'segment', 'shuttle', 'vehicle']
    
    def __init__(self, ml_name):
        self.ml_name = ml_name
        log_file = LogFile()
        self.log = log_file.get_log(self.ml_name+'_data', self.ml_name+'_data.log', Config.display_console)
        self.log_debug = log_file.get_log(self.ml_name+'_debug', self.ml_name+'_debug.log', Config.display_console)
        self.log_error = log_file.get_log(self.ml_name+'_error', self.ml_name+'_error.err', Config.display_console)
   
    def gen_ml_lst(self):
        random_lst = []
        boosted_lst = []
        bagging_lst = []
        for i in Config.base_estimation:
            random_lst.append(RandomForestClassifier(n_estimators=i))
            boosted_lst.append(GradientBoostingClassifier(n_estimators=i))
            bagging_lst.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=i))
        knn_lst = []
        
        svm_lst = [LibSVMWrapper(kernel=0),
           LibSVMWrapper(kernel=1, degree=2),
           LibSVMWrapper(kernel=1, degree=3),
           LibSVMWrapper(kernel=2),
           LibSVMWrapper(kernel=3)
           ]
    
        return {
                'bagging':bagging_lst,
                'boosted':boosted_lst,
                'randomforest':random_lst,
                'nb':[GaussianNB()],
                'knn':knn_lst,
                'decsiontree':[DecisionTreeClassifier()],
                'svm':svm_lst
        }
    
    def cross_validation(self, ml_lst, x, y):
        score_lst = []    
        for ml in ml_lst:
            self.log_debug.info('start cross val')
            try:
                scores = cross_validation.cross_val_score(ml, x, y, cv=5)
                self.log_debug.info('end cross val')
                score_lst.append(scores.mean())
            except Exception as e:
                self.log.info(str(e))       
        np_score = np.array(score_lst)
        max_idx = np_score.argmax()
        return ml_lst[max_idx]
         
    def gen_knn(self):
        knn_lst = []
        rng = range(2, 200)
        if self.ml_name == 'vehicle':
            rng = range(2, 30)

        if self.ml_name == 'heart':
            rng = range(2, 10)
                    
        for i in rng:
            knn_lst.append(KNeighborsClassifier(n_neighbors=i))
        return knn_lst
    
    def remove_by_chi2_process(self, x, y):
        from sklearn.feature_selection import  SelectKBest, f_classif
        chi2 = SelectKBest(f_classif, k=3)
        x_train = chi2.fit_transform(x, y)
        result_idx = []
        feature_len = len(x[0])
        for i in range(0, feature_len):
            column_data = x[:, i]
            if  np.array_equal(column_data, x_train[:, 0]):
                result_idx.append(i)
            if  np.array_equal(column_data, x_train[:, 1]):
                result_idx.append(i)
            if  np.array_equal(column_data, x_train[:, 2]):
                result_idx.append(i)
        x = np.delete(x, result_idx, axis=1)
        return x, y           

    def load_dataset(self):
        loader = DataSetLoader()
        lst = loader.loadData()
        return lst        
       
    def process(self):
        result = {}
        dataset_lst = self.load_dataset()
        
        self.log_debug.info('*************************************** ' + self.ml_name)
        all_data = []
        ml_lst = self.gen_ml_lst() 
        ml_value = ml_lst[self.ml_name]       
        for data_set_name in self.dataset_name_lst:
            self.log_debug.info('***** start ' + data_set_name)
            data_value = dataset_lst[data_set_name]
            x_data = data_value[0]
            y_data = data_value[1]
            print 'before************** ',x_data[0]
            x_data, y_data = self.remove_by_chi2_process(x_data, y_data)
            print 'after****************',x_data[0]
            dataset_map = {}
            datasets_data = []            
            for data_size_value in self.data_size:
                acc_lst = []
                f1_lst = []
                time_pred = []
                total_ins = []
                precision_lst = []
                recall_lst = []                
                for i in range(0, Config.reperating_loop):
                    self.log_debug.info('***** start size ' + str(data_size_value))
                    ran_num = random.randint(1, 100)
                    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=data_size_value, random_state=ran_num)
                    self.log_debug.info('********* start cross validation')
                    ml = self.cross_validation(ml_value, x_train, y_train)
                    self.log_debug.info('************* end cross validation')
                    self.log_debug.info('loop {} size {} data set {} ml {}'.format(i, data_size_value, data_set_name, self.ml_name))
                    ran_num = random.randint(1, 10000)
                    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=data_size_value, random_state=ran_num)
                    try:
                        ml_c = copy.deepcopy(ml)
                        ml_c.fit(x_train, y_train)
                        start = time.time()
                        y_pred = ml_c.predict(x_test)
                    except Exception as e:
                        self.log.info(str(e))
                    total_time = time.time() - start
                    acc = accuracy_score(y_test, y_pred)
                    fsc = f1_score(y_test, y_pred)
                    acc_lst.append(acc)
                    f1_lst.append(fsc)
                    time_pred.append(total_time)
                    total_ins.append(len(y_test))
                    pre_score = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    precision_lst.append(pre_score)
                    recall_lst.append(recall)
                    self.log_debug.info('------------- end loop -----')
                    
if __name__ == '__main__':
    ml_key = sys.argv[1]
    obj = MissingAttribute(ml_key)
    try:
        obj.process()
    except Exception as e:
        obj.log_error.info(str(e))
