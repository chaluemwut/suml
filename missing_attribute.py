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
        self.log = log_file.get_log(self.ml_name + '_data', self.ml_name + '_data.log', Config.display_console)
        self.log_debug = log_file.get_log(self.ml_name + '_debug', self.ml_name + '_debug.log', Config.display_console)
        self.log_error = log_file.get_log(self.ml_name + '_error', self.ml_name + '_error.err', Config.display_console)
   
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
        print 'start cross val -------------'
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
         
    def gen_knn(self, dataset_name):
        knn_lst = []
        rng = range(2, 200)
        if dataset_name == 'vehicle':
            rng = range(2, 30)

        if dataset_name == 'heart':
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
  
    def report_all(self, result):
        perform_result = []
        time_result = []
        for key, value in result.iteritems():
            all_data = value
            all_data = np.array(all_data)
            acc25 = np.mean(all_data[:, 0])
            f1_25 = np.mean(all_data[:, 1])
            time25 = np.mean(all_data[:, 2])
            total_ins25 = np.mean(all_data[:, 3])
            acc50 = np.mean(all_data[:, 4])
            f1_50 = np.mean(all_data[:, 5])
            time50 = np.mean(all_data[:, 6])
            total_ins50 = np.mean(all_data[:, 7])
            acc75 = np.mean(all_data[:, 8])
            f1_75 = np.mean(all_data[:, 9])
            time75 = np.mean(all_data[:, 10])
            total_ins75 = np.mean(all_data[:, 11])
            perform_result.append([key, acc25, acc50, acc75, f1_25, f1_50, f1_75])
            time_result.append([key, time25, total_ins25, time50, total_ins50, time75, total_ins75])
        self.log.info('---------- ml report ----------')
        self.log.info(tabulate(perform_result, headers=('ml name', 'acc 25', 'acc 50', 'acc 75', 'f1 25', 'f1 50', 'f1 75')))
        self.log.info('---------- time report --------')
        self.log.info(tabulate(time_result, headers=('ml name', 'time 25', 'ins 25', 'time 50', 'ins 50', 'time 75', 'ins 75')))
        self.log.info(self.report_by_dataset_v1(result))
            
    def report_by_dataset_v1(self, result):
        self.log_debug.info('report by dataset')
        result_lst = []
        for i in range(0, len(self.dataset_name_lst)):
            ml_result = []
            datasets_data = result[self.ml_name][i]
            acc25 = datasets_data[0]
            f1_25 = datasets_data[1]
            time25 = datasets_data[2]
            total_ins25 = datasets_data[3]
            acc50 = datasets_data[4]
            f1_50 = datasets_data[5]
            time50 = datasets_data[6]
            total_ins50 = datasets_data[7]
            acc75 = datasets_data[8]
            f1_75 = datasets_data[9]
            time75 = datasets_data[10]
            total_ins25 = datasets_data[11] 
            ml_result.append(self.ml_name)                               
            ml_result.append(self.dataset_name_lst[i])
            ml_result.append(acc25)
            ml_result.append(acc50)
            ml_result.append(acc75)
            ml_result.append(f1_25)
            ml_result.append(f1_50)
            ml_result.append(f1_75)
            result_lst.append(ml_result)
        return tabulate(result_lst, ('ml name', 'data set', 'acc 25', 'acc 50', 'acc 75', 'f1 25', 'f1 50', 'f1 75'))
       
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
            print 'before************** ', x_data[0]
            x_data, y_data = self.remove_by_chi2_process(x_data, y_data)
            print 'after****************', x_data[0]
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
                datasets_data.append(np.mean(acc_lst))
                datasets_data.append(np.mean(f1_lst))
                datasets_data.append(np.mean(time_pred))
                datasets_data.append(np.mean(total_ins))
                self.log.info('---------------------------------------------') 
                self.log.info('data size ' + str(data_size_value) + ' data set ' + data_set_name) 
                self.log.info(acc_lst)
                self.log.info(f1_lst)
                self.log.info(time_pred)
                self.log.info(total_ins)
                self.log.info('---------------------------------------------')
                self.log_debug.info('*********** end size')                    
            all_data.append(datasets_data)
            dataset_map[data_set_name] = all_data
            self.log_debug.info('******* end data set')
        result[self.ml_name] = all_data
        self.log_debug.info('************ end ml')
        pickle.dump(result, open(self.ml_name + '_result.obj', 'wb'))
        self.report_all(result)                    

def mainCmp(ml_key):
    print ' ---------- start cmp -------'
    print 'ml name ',ml_key
    obj = MissingAttribute(ml_key)
    obj.process()
                        
if __name__ == '__main__':
    ml_key = sys.argv[1]
    mainCmp(ml_key)
