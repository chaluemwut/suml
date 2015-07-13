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

ml_name = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree', 'svm']

class SingleProcess(object):
    data_size = [0.75, 0.50, 0.25]
    
    def __init__(self):
        log_file = LogFile()
        self.log = log_file.get_log('data', 'data.log', Config.display_console)
        self.log_debug = log_file.get_log('debug', 'debug.log', Config.display_console)
        
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
                ml_name[0]:bagging_lst,
                ml_name[1]:boosted_lst,
                ml_name[2]:random_lst,
                ml_name[3]:[GaussianNB()],
                ml_name[4]:knn_lst,
                ml_name[5]:[DecisionTreeClassifier()],
                ml_name[6]:svm_lst
        }
     
    def gen_knn(self, max_size):
        lst_random = random.sample(range(1, max_size), 10)
        knn_lst = []
        for i in lst_random:
            knn_lst.append(KNeighborsClassifier(n_neighbors=i))
        return knn_lst
        
    def load_dataset(self):
        loader = DataSetLoader()
        lst = loader.loadData()
        return lst
    
    def cross_validation(self, ml_lst, x, y):
        score_lst = []    
        for ml in ml_lst:
            self.log_debug.info('start cross val')
            scores = cross_validation.cross_val_score(ml, x, y, cv=5)
            self.log_debug.info('end cross val')
            score_lst.append(scores.mean())
        np_score = np.array(score_lst)
        max_idx = np_score.argmax()
        return ml_lst[max_idx]
    
    def find_mean_acc(self, data_value):
        lst_acc = []
        for data in data_value:
            lst_acc.append(data.acc)
        return np.mean(lst_acc)
    
    def report_all(self, result):
        lst_data = []
        time_data = []
        for m in ml_name:
            all_data = result[m]
            all_data = np.array(all_data)
            acc25 = all_data[:, 0]
            f1_25 = all_data[:, 1]
            time25 = all_data[:, 2]
            total_ins25 = all_data[:, 3]
            acc50 = all_data[:, 4]
            f1_50 = all_data[:, 5]
            time50 = all_data[:, 6]
            total_ins50 = all_data[:, 7]
            acc75 = all_data[:, 8]
            f1_75 = all_data[:, 9]
            time75 = all_data[:, 10]
            total_ins75 = all_data[:, 11]                        
            lst_data.append([m, acc25, acc50, acc75, f1_25, f1_50, f1_75])
            time_data.append([m, time25, total_ins25, time50, total_ins50, time75, total_ins75]) 
        self.log.info(tabulate(lst_data, headers=('ml name', 'acc 25', 'acc 50', 'acc 75', 'f1 25', 'f1 50', 'f1 75')))
        self.log.info('----------------------------')
        self.log.info(tabulate(time_data, headers=('ml name', 'time 25', 'ins 25', 'time 50', 'ins 50', 'time 75', 'ins 75')))
    
    def report_by_dataset_v1(self, result):
        self.log_debug.info('report by dataset')
        result_lst = []
        for i in range(0, len(DataSetLoader.dataset_name)):
            self.log_debug('----- data set ' + DataSetLoader.dataset_name[i])
            for m in ml_name:
                ml_result = []
                datasets_data = result[m][i]
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
                ml_result.append(m)
                ml_result.append(acc25)
                ml_result.append(acc50)
                ml_result.append(acc75)
                ml_result.append(f1_25)
                ml_result.append(f1_50)
                ml_result.append(f1_75)
                result_lst.append(ml_result)
        self.log.info(tabulate(result_lst, ('ml name', 'acc 25', 'acc 50', 'acc 75', 'f1 25', 'f1 50', 'f1 75')))
                        
    def report(self, result):
        self.report_by_dataset_v1(result)
        self.report_all(result)
        
    def process(self):
        ml_lst = self.gen_ml_lst()
        dataset_lst = self.load_dataset()
        result = {}
        for ml_key, ml_value in ml_lst.iteritems():
            self.log_debug.info('*************************************** ' + ml_key)
            all_data = []           
            for dataset_name in DataSetLoader.dataset_name:
                self.log_debug.info('***** start ' + dataset_name)
                data_value = dataset_lst[dataset_name]
                x_data = data_value[0]
                y_data = data_value[1]
                datasets_data = []            
                for d_size in self.data_size:
                    self.log_debug.info('***** start size ' + str(d_size))
                    ran_num = random.randint(1, 100)
                    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=d_size, random_state=ran_num)
                    self.log_debug.info('********* start cross validation')
                    if ml_key == ml_name[4]:
                        max_knn = int(len(x_train) / 5.0)
                        knn_lst = self.gen_knn(max_knn)
                        ml = self.cross_validation(knn_lst, x_train, y_train)
                    else:
                        ml = self.cross_validation(ml_value, x_train, y_train)
                    self.log_debug.info('************* end cross validation')
                    acc_lst = []
                    f1_lst = []
                    time_pred = []
                    total_ins = []
                    precision_lst = []
                    recall_lst = []
                    for i in range(0, Config.reperating_loop):
                        self.log_debug.info('loop {} size {} data set {} ml {}'.format(i, d_size, dataset_name, ml_key))
                        ran_num = random.randint(1, 10000)
                        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=d_size, random_state=ran_num)
                        ml_c = copy.deepcopy(ml)
                        ml_c.fit(x_train, y_train)
                        start = time.time()
                        y_pred = ml_c.predict(x_test)
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
                    self.log.info('data size ' + str(d_size) + ' data set ' + dataset_name) 
                    self.log.info(acc_lst)
                    self.log.info(f1_lst)
                    self.log.info(time_pred)
                    self.log.info(total_ins)
                    self.log.info('---------------------------------------------')
                    self.log_debug.info('*********** end size')                    
                all_data.append(datasets_data)
                self.log_debug.info('******* end data set')
            result[ml_key] = all_data
            self.log_debug.info('************ end ml')
        pickle.dump(result, open('result.obj', 'wb'))
        self.report_all(result)
          
        
def mainCmp():
    print ' ---------- start cmp -------'
    obj = SingleProcess()
    obj.process()
    print ' ---------- end cmp -------'
    
if __name__ == '__main__':
    mainCmp()
