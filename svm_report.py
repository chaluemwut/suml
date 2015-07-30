import pickle
import numpy as np
from tabulate import tabulate

# dataset_name = ['adult','heart', 'letter', 'austra', 'german']
dataset_name = ['sat', 'segment', 'shuttle', 'vehicle']
# dataset_name = ['sat','vehicle']

def report_by_dataset_v1(result):
    print 'report by dataset'
    result_lst = []
    for key, value in result.iteritems():
        for i in range(0, len(dataset_name)):
            ml_result = []
            datasets_data = result[key][i]
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
            ml_result.append(key)                               
            ml_result.append(dataset_name[i])
            ml_result.append(acc25)
            ml_result.append(acc50)
            ml_result.append(acc75)
            ml_result.append(f1_25)
            ml_result.append(f1_50)
            ml_result.append(f1_75)
            result_lst.append(ml_result)
    print tabulate(result_lst, ('ml name','data set','acc 25', 'acc 50', 'acc 75', 'f1 25', 'f1 50', 'f1 75'))

def report_by_dataset_v2(result, dataset_lst):
    print 'report by dataset'
    result_lst = []
    for key, value in result.iteritems():
        for i in range(0, len(dataset_name)):
            ml_result = []
            datasets_data = result[key][i]
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
            ml_result.append(key)                               
            ml_result.append(dataset_name[i])
            ml_result.append(acc25)
            ml_result.append(acc50)
            ml_result.append(acc75)
            ml_result.append(f1_25)
            ml_result.append(f1_50)
            ml_result.append(f1_75)
            result_lst.append(ml_result)
    print tabulate(result_lst, ('ml name','data set','acc 25', 'acc 50', 'acc 75', 'f1 25', 'f1 50', 'f1 75'))

def report_all(result):
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
    print '---------- ml report ----------'
    print tabulate(perform_result, headers=('ml name', 'acc 25', 'acc 50', 'acc 75', 'f1 25', 'f1 50', 'f1 75'))
    print '---------- time report --------'
    print tabulate(time_result, headers=('ml name', 'time 25', 'ins 25', 'time 50', 'ins 50', 'time 75', 'ins 75'))
    report_by_dataset_v1(result)

def report():
    result = {}
    lst = []
    ml_name = 'bagging'
    for d_name in dataset_name:
        file_name = 'result/run4/{}_result.obj'.format(ml_name)
        obj_file = pickle.load(open(file_name, 'rb'))  
        print obj_file
        data_lst = obj_file[d_name][0]
        lst.append(data_lst)
    result[ml_name] = lst
    report_all(result)

if __name__ == '__main__':
    report()
