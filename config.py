'''
Created on Jul 12, 2015

@author: off
'''

class Config(object):
#   production
    display_console = True
    base_estimation = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    reperating_loop = 50

#   Test
#     display_console = True
#     base_estimation = [2, 4]
#     reperating_loop = 2

# common
    ml_name = ['bagging', 'boosted', 'randomforest', 'nb', 'knn', 'decsiontree', 'svm']
  
