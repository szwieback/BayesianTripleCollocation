'''
Created on 11 Jun 2017

@author: zwieback
'''
import numpy as np
import os
from simulation_internal import simulation_internal
from simulation_paths import path
def Q3_simulation():
    n = 250
    nrep = 25
    niter = 2000
    nchains = 2
    
    seed = 1234
    
    scenarios = ['Q3dof', 'Q3priorfactor', 'Q3beta', 'Q3logisticspline', 'Q3ar1', 'Q3studenttsim', 'Q3studenttinference', 'Q3studenttsiminference']
    
    for scenario in scenarios:
        numpy_rng = np.random.RandomState(seed)
        
        for rep in range(nrep):
            pathoutrep = os.path.join(path, scenario, str(n), str(rep))
            simulation_internal(scenario, n, numpy_rng=numpy_rng, pathout=pathoutrep, niter=niter, nchains=nchains)
                
if __name__=='__main__':
    Q3_simulation()            