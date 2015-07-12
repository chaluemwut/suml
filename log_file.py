'''
Created on Jul 12, 2015

@author: off
'''
import logging, sys

class LogFile(object):


    def get_log(self, log_name, file_name, d_console):        
        log = logging.getLogger(log_name)
        log.setLevel(logging.DEBUG)
        format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if d_console :
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(format)
            log.addHandler(ch) 
        fh = logging.FileHandler(file_name)
        fh.setFormatter(format)
        log.addHandler(fh)
        return log
        