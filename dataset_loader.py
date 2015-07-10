import numpy as np

class DataSetLoader(object):

    def __init__(self):
        pass
    
    def template_load(self, file_name, header, data_type):
        return np.genfromtxt(file_name, dtype=data_type, delimiter=',', autostrip=True, names=header)
    
    def loadData(self):
        header=['age',
                      'workclass',
                      'fnlwgt',
                      'education',
                      'education_num',
                      'marital_status',
                      'occupation',
                      'relationship',
                      'race',
                      'sex',
                      'capital_gain',
                      'capital_loss',
                      'hours_per_week',
                      'native_country',
                      'salary']
        data_type = (int, 'S32',int, 'S32',int,'S32','S32','S32','S32','S32',int,int,int,'S32','S32')
        adult = self.template_load('data/binary/adult.txt', header, data_type)
        print adult[:,[1,2,3]]
        
#         return {'adult':[[1,2,3],[1]]}
    
if __name__ == '__main__':
    obj = DataSetLoader()
    obj.loadData()