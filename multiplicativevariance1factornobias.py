'''
Created on 14 Apr 2017

@author: Simon
'''
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import gmean
def normalized_weight(explanatory):
    # weight is assumed positive
    assert np.count_nonzero(explanatory<0)==0    
    nweight=explanatory/gmean(explanatory)
    return nweight

niter=2000
nsensors=3
seed=1234


numpy_rng = np.random.RandomState(seed)
n = 250
_a = np.array([0.0,0.1,-0.15])
_b = np.array([1.0,1.1,0.9])
_sigmapsquared=np.array([0.05,0.03,0.06])**2

_alpha1=np.array([0.1,0.5,0.8])
explan1=numpy_rng.uniform(low=0.1,high=1.0,size=n)
weight1=normalized_weight(explan1)

theta=numpy_rng.uniform(low=0,high=0.4,size=n)
noise=np.sqrt(_sigmapsquared[:,np.newaxis]*np.power(weight1[np.newaxis,:],_alpha1[:,np.newaxis]))*numpy_rng.normal(size=(nsensors,n))
y = _a[:,np.newaxis]+_b[:,np.newaxis]*theta[np.newaxis,:]+noise

if __name__=='__main__':

    with pm.Model() as model:
        # model parameterization: Bayesian Computation for Parametric Models of Heteroscedasticity in the Linear Model; Boscardin and Gelman 
        a0=tt.zeros(1)+0.0
        b0=tt.zeros(1)+1.0
        dof=4
        arest = pm.StudentT('arest', dof, mu=0, sd=0.3,shape=(nsensors-1)) # change to T
        brest = pm.StudentT('brest', dof, mu=1, sd=0.3,shape=(nsensors-1))
        
        #alpha1 = pm.Normal('alpha1', mu = 0, sd=0.05,shape=(nsensors))
        sigmapsquared=pm.Exponential('sigmapsquared', 1/0.1, shape=(nsensors))
        sigmap=pm.Deterministic('sigmap',tt.sqrt(sigmapsquared))
        alpha1=pm.StudentT('alpha1',dof,mu=0,sd=1,shape=(nsensors))
        sigmasquaredtotal=(sigmapsquared[:,np.newaxis]*tt.pow(weight1[np.newaxis,:],alpha1[:,np.newaxis]))
        
        #most basic implementation
        #theta=pm.Uniform('theta', 0, 0.4, shape=(n))#check beta formulation; with / without porosity estimate
        #estimate porosity
        #porosity = pm.StudentT('porosity',dof,mu=0.4,sd=0.1)
        #theta = pm.Uniform('theta',0,1.0, shape=(n))*porosity
        #estimate porosity and shape
        porosity = pm.StudentT('porosity',dof,mu=0.4,sd=0.1)
        alphatheta = pm.ChiSquared('alphatheta',3)
        betatheta = pm.ChiSquared('betatheta',3)
        theta = pm.Beta('theta',alpha=alphatheta,beta=betatheta, shape=(n))*porosity        
        
        y0=a0+b0*theta[np.newaxis,:]
        yrest = arest[:,np.newaxis]+brest[:,np.newaxis]*theta[np.newaxis,:] 
        
        y_est=tt.concatenate([y0,yrest],axis=0)
        
        y = pm.Normal('y', mu=y_est, sd=tt.sqrt(sigmasquaredtotal), observed=y)#sigma[:,np.newaxis]      
    
        # inference
        #trace = pm.sample(niter,random_seed=seed)
        #calling advi separately makes the whole procedure reproducible
        v_params = pm.variational.advi(n=200000, random_seed=seed)
        tracevi = pm.variational.sample_vp(v_params, draws=1000, random_seed=seed)
        step = pm.NUTS(scaling=np.power(model.dict_to_array(v_params.stds), 2), is_cov=True)
        trace = pm.sample(draws=niter, step=step, start=v_params.means,random_seed=seed)        
       

        pm.summary(trace[niter//2::],varnames=['sigmap','arest','brest','porosity','alphatheta','betatheta','alpha1'])
        #pm.traceplot(trace);
        
        plt.show()
