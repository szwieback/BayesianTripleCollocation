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

nsensors=3
n=500
# currently used for both simulation and inference
seed=1234#123

estimateexplanterms={'kappa0':True,'kappa':True,'mu':True,'mu0':True,'lambda':True,'lambda0':True,'alphabeta':False}
estimatesdexplanterms={'mu':False,'lambda':False,'kappa':False,'alphabeta':False}

# theta model options
thetamodel = 'beta'#'logistic'
# multiplicative calibration constant measures spread around thetaoffset
thetaoffset=0.15
# prior distributions
doft=4 # t distribution
dofchi=3 # chi squared for alpha/beta in beta distribution
priorfactor = 1.0 # modify default prior spreads by this factor

# other constants
softabsvalue=0.01#value for softabs function applied to modelled standard deviations etc. that should be positive

# if None, use explankappa
explanmu=None
explanlambda = None
explanalphabeta = None
spline=False
splineproperties={} if spline else None
visible,internal,normalized_weights=simulate_product(nsensors,n,spline=spline,splineproperties=splineproperties,thetaoffset=thetaoffset,seed=seed)

if __name__=='__main__':
    niter=2000
    model = model_setup(visible, normalized_weights, estimateexplanterms=estimateexplanterms, estimatesdexplanterms=estimatesdexplanterms, thetamodel=thetamodel, thetaoffset=thetaoffset, doft=doft, dofchi=dofchi, priorfactor=priorfactor, softabsvalue=softabsvalue)
    print(model.unobserved_RVs)
    trace,v_params,tracevi=model_inference(model,niter=niter,seed=seed)
    import pymc3 as pm       
    #pm.summary(tracevi,varnames=['sigmap','mest','lest','porosity','a','b','kappa'])
    #print('-------------')
    print(pm.diagnostics.gelman_rubin(trace))
    pm.summary(trace,varnames=['sigmap','mest','lest','sdmu','sdlambda','sdkappa','sdalphabeta','porosity','a','b','kappa','mu0est','muest','lambda0est','lambdaest','alpha','beta'])
    
        
