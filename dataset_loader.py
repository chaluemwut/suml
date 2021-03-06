import numpy as np

class DataSetLoader(object):
    dataset_name = ['adult', 'heart', 'letter', 'austra', 'german', 'sat', 'segment', 'shuttle', 'vehicle']

    def __init__(self):
        pass
    
    def template_load(self, file_name):
        return np.array(np.genfromtxt(file_name, dtype=int, delimiter=','))
    
    def template_load_float(self, file_name):
        return np.genfromtxt(file_name, dtype=float, delimiter=',')
    
    def reshape_y(self, y_train):
        ret = y_train.reshape(1, len(y_train))
        return ret[0]
    
    def reshape_y_convert(self, y_train):
        ret = y_train.reshape(1, len(y_train))
        return ret[0].astype(int)
        
    def get_y_last_convert(self, data):
        attr = len(data[0])
        r = range(0, attr - 2)
        x = data[:, r]
        y = self.reshape_y_convert(data[:, [attr - 1]])
        return x, y
    
    def get_y_last(self, data):
        attr = len(data[0])
        r = range(0, attr - 2)
        x = data[:, r]
        y = self.reshape_y(data[:, [attr - 1]])
        return x, y
    
    def get_y_first(self, data):
        attr = len(data[0])
        r = range(1, attr - 1)
        x = data[:, r]
        y = self.reshape_y(data[:, [0]])
        return x, y
    
    def svm_shuttle(self, data_set_name):
        shuttle = self.template_load(data_set_name)
#         uni_shuttle = np.unique(shuttle)
        x_shutt, y_shutt = self.get_y_last(shuttle)
        return {'shuttle':[x_shutt, y_shutt]}

    def load_segment(self):
        segment = self.template_load_float('data/statlog/segment.data')
        uni_seg = np.vstack({tuple(row) for row in segment})
        x_seg, y_seg = self.get_y_last_convert(uni_seg)
        return x_seg, y_seg   
           
    def loadData(self):
        adult = self.template_load('data/binary/adult.data')
        x_adult, y_adult = self.get_y_last(adult)
        
        heart = self.template_load_float('data/binary/HeartDisease.data')
        x_heart, y_heart = self.get_y_last_convert(heart)
        
        letter = self.template_load('data/binary/letter.p2.data')
        x_letter, y_letter = self.get_y_first(letter)
        
        austra = self.template_load_float('data/statlog/australian.data')
        x_aus, y_aus = self.get_y_last_convert(austra)
        
        german = self.template_load('data/statlog/german.data')
        x_ger, y_ger = self.get_y_last(german)
        
        stat = self.template_load('data/statlog/sat.data')
        x_stat, y_stat = self.get_y_last(stat)
        
        seg = self.template_load_float('data/statlog/segment.data')
        x_seg, y_seg = self.get_y_last_convert(seg)
         
        shuttle = self.template_load('data/statlog/shuttle.data')
        x_shutt, y_shutt = self.get_y_last(shuttle)
         
        vechicle = self.template_load('data/statlog/vehicle.data')
        x_vechicle, y_vechicle = self.get_y_last(vechicle)
        
        return {'adult':[x_adult, y_adult],
                'heart':[x_heart, y_heart],
                'letter':[x_letter, y_letter],
                'austra':[x_aus, y_aus],
                'german':[x_ger, y_ger],
                'sat':[x_stat, y_stat],
                'segment':[x_seg, y_seg],
                'shuttle':[x_shutt, y_shutt],
                'vehicle':[x_vechicle, y_vechicle]
                }
