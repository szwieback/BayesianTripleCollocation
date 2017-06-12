'''
Created on Jun 7, 2017

@author: zwieback
'''
import numpy as np
import os

from simulation_internal import simulation_internal
from simulation_paths import path

def Q1_simulation():
    #can we estimate additional parameters?
    ns = [100,250,500]
    nrep = 25
    niter = 2000
    nchains = 2
    
    seed = 1234
    
    scenarionames = ['Q1base','Q1lambdamu','Q1kappa','Q1spline']
    
    for scenario in scenarionames:
        for n in ns:
            numpy_rng = np.random.RandomState(seed)
            
            for rep in range(nrep):
                pathoutrep = os.path.join(path, scenario, str(n), str(rep))
                simulation_internal(scenario, n, numpy_rng=numpy_rng, pathout=pathoutrep, niter=niter, nchains=nchains)
                
if __name__=='__main__':
    Q1_simulation()     
    import Q2
    Q2.Q2_simulation()
            