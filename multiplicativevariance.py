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
    nweight=explanatory/gmean(explanatory,axis=1)[:,np.newaxis]
    return nweight

niter=2000
nsensors=3
seed=1234

numpy_rng = np.random.RandomState(seed)
n = 150
_m = np.array([0.0,0.1,-0.15])
_l = np.array([1.0,1.1,0.9])
_sigmapsquared=np.array([0.05,0.03,0.06])**2
_kappa=np.array([[0.0,0.5,0.8],[0,0,0]])#nfac, nsensors

explanmult=numpy_rng.uniform(low=0.1,high=1.0,size=(2,n))
nfacmult=explanmult.shape[0]
weightmult=normalized_weight(explanmult)

theta=numpy_rng.uniform(low=0,high=0.4,size=n)
noise=np.sqrt(_sigmapsquared[:,np.newaxis]*np.prod(np.power(weightmult[:,np.newaxis,:],_kappa[:,:,np.newaxis]),axis=0))*numpy_rng.normal(size=(nsensors,n))
y = _m[:,np.newaxis]+_l[:,np.newaxis]*theta[np.newaxis,:]+noise

estimatekappa0=False
estimatekappa=False
if __name__=='__main__':

    with pm.Model() as model:
        # model parameterization: Bayesian Computation for Parametric Models of Heteroscedasticity in the Linear Model; Boscardin and Gelman 
        m0=tt.zeros((1))+0.0
        l0=tt.zeros((1))+1.0
        dof=4
        mest = pm.StudentT('mest', dof, mu=0, sd=0.3,shape=(nsensors-1))
        lest = pm.StudentT('lest', dof, mu=1, sd=0.3,shape=(nsensors-1))
        
        sigmapsquared=pm.Exponential('sigmapsquared', 1/0.1, shape=(nsensors))
        sigmap=pm.Deterministic('sigmap',tt.sqrt(sigmapsquared))
        if estimatekappa:
            if estimatekappa0:
                kappa=pm.StudentT('kappa',dof,mu=0,sd=1,shape=(nfacmult,nsensors))
                kappaest=pm.Deterministic('kappaest',kappa)
            else:
                kappaest=pm.StudentT('kappaest',dof,mu=0,sd=1,shape=(nfacmult,nsensors-1))
                kappa0=tt.zeros((nfacmult,1))
                kappa=tt.concatenate([kappa0,kappaest],axis=1)
            sigmasquaredtotal=(sigmapsquared[:,np.newaxis]*tt.prod(tt.pow(weightmult[:,np.newaxis,:],kappa[:,:,np.newaxis]),axis=0))
        else:
            kappa = pm.Deterministic('kappa',tt.zeros((1,nsensors)))
            sigmasquaredtotal=sigmapsquared[:,np.newaxis]
        porosity = pm.StudentT('porosity',dof,mu=0.4,sd=0.1)
        alphatheta = pm.ChiSquared('alphatheta',3)
        betatheta = pm.ChiSquared('betatheta',3)
        theta = pm.Beta('theta',alpha=alphatheta,beta=betatheta, shape=(n))*porosity        
        
        y0=m0+l0*theta[np.newaxis,:]
        yrest = mest[:,np.newaxis]+lest[:,np.newaxis]*theta[np.newaxis,:] 
        
        y_est=tt.concatenate([y0,yrest],axis=0)
        
        y = pm.Normal('y', mu=y_est, sd=tt.sqrt(sigmasquaredtotal), observed=y)#sigma[:,np.newaxis]      
    
        v_params = pm.variational.advi(n=200000, random_seed=seed)
        tracevi = pm.variational.sample_vp(v_params, draws=1000, random_seed=seed)
        step = pm.NUTS(scaling=np.power(model.dict_to_array(v_params.stds), 2), is_cov=True)
        trace = pm.sample(draws=niter, step=step, start=v_params.means,random_seed=seed)        
       
        pm.summary(tracevi,varnames=['sigmap','mest','lest','porosity','alphatheta','betatheta','kappa'])
        print('-------------')
        pm.summary(trace[niter//2::],varnames=['sigmap','mest','lest','porosity','alphatheta','betatheta','kappa'])
        
        plt.show()
