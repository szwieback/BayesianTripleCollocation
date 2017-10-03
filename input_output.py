'''
Created on Jun 6, 2017

@author: zwieback
'''
import os
import dill as pickle#import pickle
def enforce_directory(fn):
    try:
        os.makedirs(os.path.dirname(fn))
    except:
        pass    
    
def save_pickle(fn, objectout):
    enforce_directory(fn)
    with open(fn,'wb') as f:
        pickle.dump(objectout, f) 
           
def read_pickle(fn):
    with open(fn,'rb') as f:
        return pickle.load(f)        
    
def save_results(pathout, resultsdict):
    for outputtype in resultsdict:
        save_pickle(os.path.join(pathout, outputtype+'.p'), resultsdict[outputtype])     
        
def read_results(pathin):
    results={}
    for outputtype in ['trace','v_params','tracevi','visible','normalized_weights','internal']:
        try:
            res=read_pickle(os.path.join(pathin, outputtype+'.p'))
            results[outputtype]=res
        except:
            #raise
            pass
    return results