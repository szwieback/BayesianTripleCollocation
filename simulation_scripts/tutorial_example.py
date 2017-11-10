'''
Created on Nov 6, 2017

@author: zwieback

This script estimates a simple error structure from simulated data.

It consists of several parts:
    - Simulation parameter definition
    - Simulation
    - Model setup
    - Inference
'''

from simulate_product import simulate_product
from model_inference import model_inference
from model_setup import model_setup

import numpy as np
import math
from scipy.interpolate import interp1d

### Simulation parameter definition ###
params={}
# 3 products
params['m'] = np.array([0.0,0.03,-0.05]) #additive bias
params['l'] = np.array([1.0,1.1,0.9]) # multiplicative bias
sigmap = np.array([0.02,0.04,0.05]) # standard deviation of quasi-random noise
params['sigmapsquared'] = sigmap**2
params['thetaoffset'] = 0.15 # multiplicatve bias is with respect to this theta value
# use antecedent precipitation index model to simulate soil moisture
# with some seasonality (not important)
params['soilmoisturemodel'] = 'smapapi' 
params['porosity'] = 0.4
params['prob_dry_model'] = lambda fy: np.array([0.55-0.1*math.cos(2*math.pi*fy), 0.85-0.1*math.cos(2*math.pi*fy)])
params['mean_rainfall_model'] = lambda fy: 0.006
params['loss_model'] = lambda fy: 0.9+0.06*math.cos(2*math.pi*fy)      
params['spline'] = False


### Simulate product ###
n = 500 #500 measurements
numpy_rng = np.random.RandomState(1) # set seed for random number generator
# visible contains the measurements; internal contains e.g. the unobservable soil moisture, and the normalized_weights the explanatory variables (empty)
visible, internal, normalized_weights = simulate_product(n, params, numpy_rng=numpy_rng)


### Model setup ###
# defines the probabilistic model
model = model_setup(visible, normalized_weights)


### Inference ###
# estimate the error parameters by conditioning on the data
# trace contains the Monte Carlo samples
trace, v_params, tracevi = model_inference(model, seed=numpy_rng.randint(0,10000))

# assess RMSE estimates
np.set_printoptions(precision=3)
print('Estimated RMSE')
print(np.squeeze(trace.get_values('sigmap').mean(axis=0))) # burn-in samples already removed
print('True RMSE')
print(sigmap)