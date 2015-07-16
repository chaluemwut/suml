import os
import numpy as np
from config import Config

libsvm_path = Config.svm_path
temp_path = 'tmp/'

class LibSVMWrapper(object):

    def __gen_str_line(self, x_data_train, y_data_train):
        data = str(y_data_train) + ' '
        counter = 1
        for i in x_data_train:
            data = data + str(counter) + ':' + str(i) + ' '
            counter = counter + 1
        return data+'\n'

    def __write_data_file(self, f_file, x_data_train, y_data_train):
        data = ''
        for x_data, y_data in zip(x_data_train, y_data_train):
            data = data+self.__gen_str_line(x_data, y_data)
        f_file.write(data)

    def __read_result(self):
        lst = []
        with open(self.path_result, 'r') as f:
            for line in f:
                lst.append(int(line[:-1]))
        return lst
    
    def __gen_random_str(self):
        lst = ['a','b','c','d','e','f','g','h','i']
        np.random.shuffle(lst)
        random = (''.join(lst))+str(np.random.random())
        return random
        
    def __gen_model_name(self):
        random = self.__gen_random_str()
        return temp_path+'{}.model'.format(random)
            
    def __init__(self, kernel=None, degree=None):
        self.kernel = kernel
        self.degree = degree
        lst = ['a','b','c','d','e','f','g','h','i']
        np.random.shuffle(lst)
        random = (''.join(lst))+str(np.random.random())
        self.path_model_file = temp_path+'{}'.format(random)
        self.path_test_data = temp_path+'{}.test'.format(random)
        self.path_result = temp_path+'{}.result'.format(random)

    def fit(self, x, y):
        self.path_model_result = self.__gen_model_name()
        f_file = open(self.path_model_file, 'w')
        self.__write_data_file(f_file, x, y)
        create_model = libsvm_path+'/svm-train'
        if self.kernel != None:
            create_model = create_model+' -t '+str(self.kernel)
        if self.degree != None:
            create_model = create_model+' -d '+str(self.degree)
        create_model = create_model+' {} {}'.format(self.path_model_file,
                                                    self.path_model_result)
        os.system(create_model)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def predict(self, x):
        f_result = open(self.path_test_data, 'w')
        self.__write_data_file(f_result, x, [0]*len(x))
        f_result.close()
        create_predict = libsvm_path+'/svm-predict'+' {} {} {}'.format(self.path_test_data,
                                                                    self.path_model_result,
                                                                    self.path_result)
        os.system(create_predict)
        return self.__read_result()

    def get_params(self, deep=True):
        return {"kernel": self.kernel}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
if __name__ == '__main__':
    from sklearn.metrics import accuracy_score
    from dataset_loader import DataSetLoader
    from sklearn import cross_validation
    loader = DataSetLoader()
    x, y = loader.loadData()[DataSetLoader.dataset_name[1]]
    lst = [LibSVMWrapper(kernel=0),
           LibSVMWrapper(kernel=1, degree=2),
           LibSVMWrapper(kernel=1, degree=3),
           LibSVMWrapper(kernel=2),
           LibSVMWrapper(kernel=3)
           ]
    from sklearn import linear_model
#     clf = linear_model.SGDClassifier()
#     clf.fit(x,y)
#     print clf.predict([x[0], x[1]])
#     scores = cross_validation.cross_val_score(clf, x, y, cv=5)
#     print scores
    from sklearn import cross_validation
    ml = LibSVMWrapper(kernel=0)
    from ml_util import MLUtil
    sc = MLUtil.cross_validation(ml, x, y, cv=5)
    print sc