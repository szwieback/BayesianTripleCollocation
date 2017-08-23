'''
Created on Jul 4, 2017

@author: zwieback
'''
from input_output import read_results
import numpy as np
from collections import namedtuple
import glob, os

parameter_summary = namedtuple('parameter_summary', ['trueValue','estimatedValue','estimatedUncertainty']) 

def clip(arrin, clipfirstvalue, parameter = None):
    arrout = np.array(arrin)
    if clipfirstvalue:
        arrout = arrout[:,1:,...]
    return arrout

def extract_values(param, trace, internal):
    if param == 'sigmap':
        valtrue = np.sqrt(internal['params']['sigmapsquared'])
    else:
        valtrue = internal['params'][param]
    valtrue = np.squeeze(valtrue)
    
    if param in ['lambda','mu']:                
        valtrace = np.concatenate((trace.get_values(param+'0est')[...,np.newaxis],trace.get_values(param+'est')),axis=2)
    elif param in ['m','l']:
        try:            
            valtrace = np.concatenate((trace.get_values(param+'0')[...,np.newaxis],trace.get_values(param+'est')),axis=1)
        except:
            valtrace = trace.get_values(param+'est')
            valtrace = np.concatenate((np.zeros(valtrace.shape[0])[...,np.newaxis]+np.nan,valtrace),axis=1)        
    else:
        valtrace = trace.get_values(param)    
    return valtrue, valtrace

def results_simulation_path(pathrep, recursive = False):
    if recursive:
        pathsout = glob.glob(os.path.join(pathrep,'*'), recursive = True)
        results = []
        for pathi in pathsout:
            try:
                res = results_simulation_path(pathi)
                results.append(res)
            except:
                pass
        return results
    else:
        result = read_results(pathrep)
        return result

def analyse_simulation_path(pathrep, params = ['sigmap', 'kappa', 'mu', 'lambda'], recursive = False):
    if recursive:
        pathsout = glob.glob(os.path.join(pathrep,'*'), recursive = True)
        summaries = []
        for pathi in pathsout:
            try:
                summary = analyse_simulation_path(pathi, params=params)
                summaries.append(summary)
            except:
                pass
        return summaries
    else:
        res = read_results(pathrep)
        trace = res['trace']
        internal = res['internal']
        summary = {}
        for param in params:
            valtrue, valtrace = extract_values(param, trace, internal)
            valest = np.squeeze(valtrace.mean(axis=0))
            valeststd = np.squeeze(valtrace.std(axis=0))
            summary[param] = parameter_summary(valtrue, valest, valeststd)
        return summary

def compute_metrics(summary, parameter = None, clipfirstvalue = []):#clip keyword
    # summary must be list
    if parameter is None or isinstance(parameter, list):
        pass
    else:       
        devrel = clip([(sumind[parameter].estimatedValue-sumind[parameter].trueValue)/sumind[parameter].trueValue for sumind in summary], clipfirstvalue, parameter = parameter)        
        relbiasmag = np.sqrt(np.mean(np.mean(devrel,axis=0)**2))
        relunctpost = np.mean(clip([(sumind[parameter].estimatedUncertainty/np.abs(sumind[parameter].trueValue)) for sumind in summary], clipfirstvalue, parameter = parameter))
        relunctemp = np.sqrt(np.mean(devrel**2))
        
        dev = clip([sumind[parameter].estimatedValue-sumind[parameter].trueValue for sumind in summary], clipfirstvalue, parameter = parameter)
        absbias = np.sqrt(np.mean(np.mean(dev,axis=0)**2))
        absuncpost = np.mean(clip([sumind[parameter].estimatedUncertainty for sumind in summary], clipfirstvalue, parameter = parameter))
        absunctemp = np.sqrt(np.mean(dev**2))
        
        return {'relative_bias_magnitude':relbiasmag, 'relative_uncertainty_posterior': relunctpost, 'relative_uncertainty_empirical': relunctemp,
                'absolute_bias_magnitude':absbias, 'absolute_uncertainty_posterior': absuncpost, 'absolute_uncertainty_empirical': absunctemp}
        
if __name__=='__main__':
    pathout='C:\\Work\\SMAP\\simulations\\Q3studenttinference\\500\\'
    summary = analyse_simulation_path(pathout, recursive=True)
    print(np.mean([summary0['sigmap'].estimatedValue[2] for summary0 in summary]))
    #print(compute_metrics(summary, parameter = 'sigmap', clipfirstvalue = True))
    