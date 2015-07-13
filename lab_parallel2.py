from multiprocessing import Pool
import time
 
class C:        
    def f(self, name):
        print 'hello %s,'%name
        time.sleep(50)
        print 'nice to meet you.'
     
    def run(self):
        pool = Pool(processes=10)
        pool.map(self.f, ('frank', 'justin', 'osi', 'thomas'))
 
if __name__ == '__main__':
    c = C()
    c.run()