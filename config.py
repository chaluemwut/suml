'''
Created on Jul 12, 2015

@author: off 
'''

class Config(object):
<<<<<<< HEAD
#     svm_path = '/home/off/libsvm-3.20'
    display_console = True
=======
    svm_path = '/home/ecp/program/libsvm-3.20'
#     svm_path = '/home/ubuntu/libsvm-3.20'
>>>>>>> 18ae3fe9ebdf53b1181435579ea1df336f31f97f
    
#   production
#     svm_path = '/home/ecp/program/libsvm-3.20'
#     base_estimation = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#     reperating_loop = 50

#   Test
    base_estimation = [2, 4]
    reperating_loop = 2
    svm_path = '/home/off/libsvm-3.20'

# common
    ml_name = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree', 'svm']
  
