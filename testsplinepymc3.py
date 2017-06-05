'''
Created on 4 Jun 2017

@author: Simon
'''
import pymc3 as pm
import theano.tensor as tt
from theano import shared
from spline_predictors import periodic_spline_predictors
import numpy as np
import pylab as plt

n=150
nsegments=12

fy=np.random.rand(n)
fy=np.sort(fy)
y=np.cos(2*fy*(2*np.pi)+1.1)+np.random.randn(n)*0.3
psp=periodic_spline_predictors(fy,nsegments=nsegments)
basisf_=psp['basisfunctions']['diffcoeff']
basisf = shared(basisf_)

with pm.Model() as model:
    sigma_diffcoeff = pm.Exponential('sigma_diffcoeff', 1.0)
    meanval = pm.Normal('meanval', 0.0, 1.0)
    diffcoeff = pm.Normal('diffcoeff', 0.0, sd=sigma_diffcoeff, shape=nsegments)
    
    sigma = pm.Exponential('sigma', 1.0)
    
    obs = pm.Normal('obs', basisf.dot(diffcoeff)+meanval, sigma, observed=y)
with model:
    trace = pm.sample(1000)
    pm.summary(trace,varnames=['diffcoeff','sigma'])
trace=trace[500::]
diffcoeff_est=np.mean(trace['diffcoeff'],axis=0)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(fy,y)
ax.plot(fy,psp['evaluation']['diffcoeff'](diffcoeff_est)+np.mean(trace['meanval']))
plt.show()
