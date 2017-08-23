'''
Created on Jun 8, 2017

@author: zwieback
'''
import numpy as np
import os
from simulation_internal import simulation_internal
from simulation_paths import path
def Q2_simulation():
    ns = [100,250,500]
    nrep = 25
    niter = 2000
    nchains = 2
    
    seed = 1234
    
    scenarios = ['Q2kappa_base', 'Q2lambdamu_base', 'Q2kappa_lambdamu', 'Q2spline_base']
    
    for scenario in scenarios:
        for n in ns:
            numpy_rng = np.random.RandomState(seed)
            
            for rep in range(nrep):
                pathoutrep = os.path.join(path, scenario, str(n), str(rep))
                simulation_internal(scenario, n, numpy_rng=numpy_rng, pathout=pathoutrep, niter=niter, nchains=nchains)
                
if __name__=='__main__':
    Q2_simulation()                   