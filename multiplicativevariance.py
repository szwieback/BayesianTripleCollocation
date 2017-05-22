'''
Created on 14 Apr 2017

@author: Simon
'''
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import gmean
def normalized_weight_multiplicative(explanatory):
    # weight is assumed positive
    assert np.count_nonzero(explanatory<0)==0    
    nweight=explanatory/gmean(explanatory,axis=1)[:,np.newaxis]
    return nweight

def normalized_weight_additive(explanatory):
    weight = explanatory-np.mean(explanatory,axis=1)[:,np.newaxis]
    nweight = weight / np.std(weight,axis=1)[:,np.newaxis]
    return nweight

niter=2000
nsensors=3
seed=12#1234

estimatekappa0=False
estimatekappa=True
estimatemu=True
estimatelambda=True

explanmu=None
explanlambda = None

thetaoffset=0.15

numpy_rng = np.random.RandomState(seed)
n = 200
_m = np.array([0.0,0.1,-0.15])
_l = np.array([1.0,1.1,0.9])#0.9
_sigmapsquared=np.array([0.05,0.03,0.06])**2
_kappa=np.array([[0.0,0.0,0.0],[0,0,0]])#nfac, nsensors
_mu=np.array([[0.0,0.00,0.0],[0,0,0]])
_lambda=np.array([[0.0,0.00,0.0],[0,0,0]])

explankappa=numpy_rng.uniform(low=0.1,high=1.0,size=(2,n))
if explankappa is None:
    explankappa = np.zeros((0,n))
nfackappa=explankappa.shape[0]
weightkappa=normalized_weight_multiplicative(explankappa)

if explanmu is None:
    explanmu = explankappa
nfacmu = explanmu.shape[0]
weightmu=normalized_weight_additive(explanmu)
if explanlambda is None:
    explanlambda = explankappa
nfaclambda = explanlambda.shape[0]
weightlambda = normalized_weight_additive(explanlambda)

theta=numpy_rng.uniform(low=0,high=0.4,size=n)
_sigmasquared = _sigmapsquared[:,np.newaxis]*np.prod(np.power(weightkappa[:,np.newaxis,:],_kappa[:,:,np.newaxis]),axis=0)
#_sigmasquared = _sigmapsquared[:,np.newaxis]
noise=np.sqrt(_sigmasquared)*numpy_rng.normal(size=(nsensors,n))
_M=_m[:,np.newaxis]+np.sum(_mu[:,:,np.newaxis]*weightmu[:,np.newaxis,:],axis=0)
_L=_l[:,np.newaxis]+np.sum(_lambda[:,:,np.newaxis]*weightlambda[:,np.newaxis,:],axis=0)
#_M=_m[:,np.newaxis]
#_L=_l[:,np.newaxis]
y = _M+_L*(theta[np.newaxis,:]-thetaoffset)+noise

if __name__=='__main__':

    with pm.Model() as model:
        # model parameterization: Bayesian Computation for Parametric Models of Heteroscedasticity in the Linear Model; Boscardin and Gelman 
        m0=tt.zeros((1))+0.0
        l0=tt.zeros((1))+1.0
        dof=4
        mest = pm.StudentT('mest', dof, mu=0, sd=0.3,shape=(nsensors-1))
        lest = pm.StudentT('lest', dof, mu=1, sd=0.3,shape=(nsensors-1))
        
        if estimatemu:
            muest = pm.StudentT('muest',dof,mu=0,sd=0.1,shape=(nfacmu,nsensors-1))
            Mest = mest[:,np.newaxis] + tt.sum(muest[:,:,np.newaxis]*weightmu[:,np.newaxis,:],axis=0)
        else:
            muest = pm.Deterministic('muest',tt.zeros((nfacmu,nsensors-1)))
            Mest = mest[:,np.newaxis]
        
        if estimatelambda:
            lambdaest = pm.StudentT('lambdaest',dof,mu=0,sd=0.1,shape=(nfaclambda,nsensors-1))
            Lest = lest[:,np.newaxis] + tt.sum(lambdaest[:,:,np.newaxis]*weightlambda[:,np.newaxis,:],axis=0)
        else:
            lambdaest = pm.Deterministic('lambdaest',tt.zeros((nfaclambda,nsensors-1)))
            Lest = lest[:,np.newaxis]            
        
        sigmapsquared=pm.Exponential('sigmapsquared', 1/0.1, shape=(nsensors))
        sigmap=pm.Deterministic('sigmap',tt.sqrt(sigmapsquared))
        if estimatekappa:
            if estimatekappa0:
                kappa=pm.StudentT('kappa',dof,mu=0,sd=1,shape=(nfackappa,nsensors))
                kappaest=pm.Deterministic('kappaest',kappa)
            else:
                kappaest=pm.StudentT('kappaest',dof,mu=0,sd=1,shape=(nfackappa,nsensors-1))
                kappa0=tt.zeros((nfackappa,1))
                kappa=pm.Deterministic('kappa',tt.concatenate([kappa0,kappaest],axis=1))
            sigmasquaredtotal=(sigmapsquared[:,np.newaxis]*tt.prod(tt.pow(weightkappa[:,np.newaxis,:],kappa[:,:,np.newaxis]),axis=0))
        else:
            kappa = pm.Deterministic('kappa',tt.zeros((1,nsensors)))
            sigmasquaredtotal=sigmapsquared[:,np.newaxis]
            
        porosity = pm.StudentT('porosity',dof,mu=0.4,sd=0.1)
        alphatheta = pm.ChiSquared('alphatheta',3)
        betatheta = pm.ChiSquared('betatheta',3)
        theta = pm.Beta('theta',alpha=alphatheta,beta=betatheta, shape=(n))*porosity        
        
        y0=m0+l0*(theta[np.newaxis,:]-thetaoffset)
        yrest = Mest+Lest*(theta[np.newaxis,:]-thetaoffset) 
        
        yest=tt.concatenate([y0,yrest],axis=0)
        
        y = pm.Normal('y', mu=yest, sd=tt.sqrt(sigmasquaredtotal), observed=y)#sigma[:,np.newaxis]      
    
        v_params = pm.variational.advi(n=200000, random_seed=seed)
        tracevi = pm.variational.sample_vp(v_params, draws=1000, random_seed=seed)
        step = pm.NUTS(scaling=np.power(model.dict_to_array(v_params.stds), 2), is_cov=True)
        trace = pm.sample(draws=niter, step=step, start=v_params.means,random_seed=seed)        
       
        pm.summary(tracevi,varnames=['sigmap','mest','lest','porosity','alphatheta','betatheta','kappa'])
        print('-------------')
        pm.summary(trace[niter//2::],varnames=['sigmap','mest','lest','porosity','alphatheta','betatheta','kappa','muest','lambdaest'])
        
        plt.show()
