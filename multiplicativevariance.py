'''
Created on 14 Apr 2017

@author: Simon
'''
import numpy as np
#import matplotlib.pyplot as plt
#import pymc3 as pm
from model_setup import model_setup
from model_inference import model_inference
from simulate_product import simulate_product

nsensors = 3
n = 250
# currently used for both simulation and inference
seed = 1234#123

estimateexplanterms = {'kappa0':False, 'kappa':False, 'mu':True, 'mu0':False, 'lambda':False, 'lambda0':False, 'alphabeta':False, 'sigmap0': False}
estimatesdexplanterms = {'mu':False, 'lambda':False, 'kappa':False, 'alphabeta':False}

# theta model options
thetamodel = 'logistic'#'beta'; logistic much faster
thetaoffset = 0.15


# if None, use explankappa
explanmu = None
explanlambda = None
explanalphabeta = None
soilmoisturemodel = 'smapapi'

nrepetitions=1
niter=2000

numpy_rng = np.random.RandomState(seed)
scenario='test2'
if __name__=='__main__':
    from input_output import save_results, read_results
    import os
    pathout='C:\\Work\\SMAP'
    params={}
    params['m'] = np.array([0.0,0.03,-0.05]) #nsensors;  intercept of additive calibration constant
    params['l'] = np.array([1.0,1.1,0.9]) # nsensors; intercept of multiplicative calibration constant
    params['sigmapsquared'] = np.array([0.02,0.04,0.06])**2
    params['thetaoffset'] = thetaoffset
    
    # prior distributions
    doft = 4 # t distribution
    dofchi = 3 # chi squared for alpha/beta in beta distribution
    priorfactor = 1.0 # modify default prior spreads by this factor

    # other constants
    softabsvalue = 0.01#value for softabs function applied to modelled standard deviations etc. that should be positive
    studenterrors = False
    
    inferenceparams={'thetaoffset':0.15,'thetamodel':'logistic', 'doft':4, 'dofchi':3, 'priorfactor':1.0, 'softabsvalue':0.01, 'studenterrors': False}

    
    
    for rep in range(nrepetitions):
        
        pathoutrep = os.path.join(pathout, scenario, str(rep))
        
        visible, internal, normalized_weights = simulate_product(n, params, numpy_rng=numpy_rng)
        model = model_setup(visible, normalized_weights, estimateexplanterms=estimateexplanterms, 
                            estimatesdexplanterms=estimatesdexplanterms, inferenceparams=inferenceparams)
        #print([(x,x.tag.test_value.shape) for x in model.unobserved_RVs])
        trace, v_params, tracevi = model_inference(model, niter=niter, seed=numpy_rng.randint(0,10000))
        save_results(pathoutrep, {'trace':trace,'v_params':v_params,'visible':visible,'normalized_weights':normalized_weights})
        print(visible['explan'])
        import pymc3 as pm
        #pm.summary(tracevi,varnames=['sigmap','mest','lest','porosity','a','b','kappa'])
        print('-------------')
        #print(pm.diagnostics.gelman_rubin(trace))
        trace = read_results(pathoutrep)['trace']
        #pm.summary(trace,varnames=['sigmap','mest','lest','sdmu','sdlambda','sdkappa','sdalphabeta','porosity','a','b','kappa','mu0est','muest','lambda0est','lambdaest','alpha','beta'])
        pm.summary(trace,varnames=['mest','lest','sigmap'])
