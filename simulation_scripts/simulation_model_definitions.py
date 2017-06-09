'''
Created on Jun 7, 2017

@author: zwieback
'''
import numpy as np
import math
from scipy.interpolate import interp1d
def simulation_model(scenarioname):
    params={}
    Q1fyp = np.linspace(0,1,13)
    Q1weightp = np.array([0,0,0,1,2,3,3,1,0,0,0,0,0])
    Q1explanmodel = lambda fy: interp1d(Q1fyp,Q1weightp)(fy)[np.newaxis,:]
    Q1weightpkappa = np.array([1,1,1,1,2,3,3,1,1,1,1,1,1])
    Q1explanmodelkappa = lambda fy: interp1d(Q1fyp,Q1weightpkappa)(fy)[np.newaxis,:]
    if scenarioname == 'Q1base':
        params['m'] = np.array([0.0,0.03,-0.05]) #nsensors;  intercept of additive calibration constant
        params['l'] = np.array([1.0,1.1,0.9]) # nsensors; intercept of multiplicative calibration constant
        params['sigmapsquared'] = np.array([0.02,0.04,0.05])**2
        params['thetaoffset'] = 0.15
        params['soilmoisturemodel'] = 'smapapi'
        params['porosity'] = 0.4
        params['prob_dry_model'] = lambda fy: np.array([0.55-0.1*math.cos(2*math.pi*fy), 0.85-0.1*math.cos(2*math.pi*fy)])
        params['mean_rainfall_model'] = lambda fy: 0.006
        params['loss_model'] = lambda fy: 0.9+0.06*math.cos(2*math.pi*fy)      
        params['spline'] = False
    elif scenarioname == 'Q1lambdamu':        
        params = simulation_model('Q1base')
        params['explanmodellambda'] = params['explanmodelmu'] = Q1explanmodel
        params['lambda'] = np.array([0.0,0.1,0.0])[np.newaxis,:]
        params['mu'] = np.array([0.0,0.05,-0.05])[np.newaxis,:]
    elif scenarioname == 'Q1kappa':
        params = simulation_model('Q1lambdamu')
        params['explanmodelkappa'] = Q1explanmodelkappa
        params['kappa'] = np.array([0.0,0.5,-0.5])[np.newaxis,:]
    elif scenarioname == 'Q1spline':
        params = simulation_model('Q1lambdamu')
    elif scenarioname == 'Q2lambdamu_base':
        params = simulation_model('Q1lambdamu')
    elif scenarioname in ['Q2kappa_lambdamu', 'Q2kappa_base']:
        params = simulation_model('Q1kappa')  
    else:
        raise NotImplementedError
    return params