from multiprocessing import Pool
import time
         
def f(name):
    print 'hello %s,'%name
    for i in xrange(100000000000):
        for j in xrange(100000):
            print i
    print 'nice to meet you.'
 
def run():
    pool = Pool(processes=10)
    pool.map(f, ('frank', 'justin', 'osi', 'thomas'))
 
if __name__ == '__main__':
    run()