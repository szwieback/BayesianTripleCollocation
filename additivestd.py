'''
Created on 14 Apr 2017

@author: Simon
'''
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor

niter=1000

nsensors=3

n = 200
_b = np.array([1.0,1.1,0.9])
_a = np.array([0.0,0.1,-0.15])
_sigma=np.array([0.05,0.03,0.06])
_alpha1=np.array([0.01,0.02,0.01])
explan1=np.random.uniform(low=-1.0,high=1.0,size=n)
theta=np.random.uniform(low=0,high=0.4,size=n)
noise=(_sigma[:,np.newaxis]+_alpha1[:,np.newaxis]*explan1[np.newaxis,:])*np.random.normal(size=(nsensors,n))
y = _a[:,np.newaxis]+_b[:,np.newaxis]*theta[np.newaxis,:]+noise

if __name__=='__main__':

    with pm.Model() as model:
        # model parameterization: Bayesian Computation for Parametric Models of Heteroscedasticity in the Linear Model; Boscardin and Gelman 
        a0=theano.tensor.zeros(1)+0.0
        b0=theano.tensor.zeros(1)+1.0
        arest = pm.Normal('arest', mu=0, sd=0.3,shape=(nsensors-1))
        brest = pm.Normal('brest', mu=1, sd=0.3,shape=(nsensors-1))
        
        alpha1 = pm.Normal('alpha1', mu = 0, sd=0.05,shape=(nsensors))
        sigma0=pm.Uniform('sigma', 0, 0.5, shape=(nsensors))
        #sigma=sigma0[:,np.newaxis]*theano.tensor.ones((nsensors,n))
        
        #check good model formulations
        #estimate white noise, modulate by model for stdev (no problem with negative numbers)
        
        #sigmatotal=sigma+alpha1[:,np.newaxis]*explan1[np.newaxis,:]
        sigmatotal=(sigma0[:,np.newaxis]+alpha1[:,np.newaxis]*explan1[np.newaxis,:])
        sigmatotalfiltered=sigmatotal#theano.tensor.abs_(sigmatotal)
        
        theta=pm.Uniform('theta', 0, 0.4, shape=(n))
        
        y0=a0+b0*theta[np.newaxis,:]
        yrest = arest[:,np.newaxis]+brest[:,np.newaxis]*theta[np.newaxis,:] 
        
        y_est=theano.tensor.concatenate([y0,yrest],axis=0)
        
        y = pm.Normal('y', mu=y_est, sd=sigmatotalfiltered, observed=y)#sigma[:,np.newaxis]
        #no alternative formulation with sigma * unit normal because deterministic has no observed keyword
        
    
        # inference
        #start = pm.find_MAP()
        step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
        #trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
        trace = pm.sample(niter,random_seed=123)
        
        
        #v_params = pm.variational.advi(n=100000)
        #trace = pm.variational.sample_vp(v_params, draws=5000)
        pm.summary(trace[niter//2::],varnames=['sigma','arest','brest','alpha1'])
        #pm.traceplot(trace);
        
        plt.show()
