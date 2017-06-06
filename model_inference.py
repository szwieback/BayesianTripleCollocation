'''
Created on Jun 1, 2017

@author: zwieback
'''
import pymc3 as pm
import numpy as np
from pymc3.backends.base import merge_traces
def model_inference(model,niter=2000,nadvi=200000,ntraceadvi=1000,seed=123,nchains=2):
    with model:
        # inference
        v_params = pm.variational.advi(n=nadvi, random_seed=seed)
        tracevi = pm.variational.sample_vp(v_params, draws=ntraceadvi, random_seed=seed)
        #trace = pm.sample(draws=niter, step=step, njobs=2, start=v_params.means,random_seed=seed)
        traces=[]
        for chain in range(nchains):
            #step = pm.NUTS(scaling=np.power(model.dict_to_array(v_params.stds), 2), is_cov=True)                        
            step = pm.NUTS(scaling=np.power(model.dict_to_array(v_params.stds), 2), is_cov=True)
            trace = pm.sample(niter, chain=chain, step=step,random_seed=seed)#start=tracevi[chain]
            trace=trace[niter//2::2]
            traces.append(trace)
        trace=merge_traces(traces)
    return trace,v_params,tracevi