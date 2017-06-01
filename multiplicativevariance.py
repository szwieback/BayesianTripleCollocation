'''
Created on 14 Apr 2017

@author: Simon
'''
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import gmean

# normalize explanatory variables such that their product is 1 (i.e. divide by their geometric mean)
def normalized_weight_multiplicative(explanatory):
    # weight is assumed positive
    assert np.count_nonzero(explanatory<0)==0
    multfactor=gmean(explanatory,axis=1)    
    nweight=explanatory/multfactor[:,np.newaxis]
    return nweight, {'multiplicative':multfactor}

# normalize explanatory variables so that they have mean zero and standard deviation 1
def normalized_weight_additive(explanatory):
    addfactor = np.mean(explanatory,axis=1)
    weight = explanatory-addfactor[:,np.newaxis]
    multfactor=np.std(weight,axis=1)
    nweight = weight / multfactor[:,np.newaxis]
    return nweight, {'multiplicative':multfactor,'additive':addfactor}


nsensors=3
# for inference
niter=2000
# currently used for both simulation and inference
seed=123#1234

estimatekappa0=False
estimatekappa=False
estimatemu=True
estimatelambda=False
estimatemu0=True
estimatelambda0=False

estimatealphabeta=False # explanatory coefficients for alpha model 

estimatesdmu=True
estimatesdlambda=False
estimatesdkappa=False
estimatesdalphabeta=False

# if None, use explankappa
explanmu=None
explanlambda = None
explanalphabeta = None

# theta model options
thetamodel = 'logistic'

# multiplicative calibration constant measures spread around thetaoffset
thetaoffset=0.15
# dof of prior distributions
dof=4 # t distribution
dofchi=3 # chi squared for alpha/beta in beta distribution
softabsvalue=0.01#value for softabs function applied to modelled standard deviations etc. that should be positive
# initialize random number generator
numpy_rng = np.random.RandomState(seed)

# number of observations
n = 200

_m = np.array([0.0,0.1,-0.15]) #nsensors;  intercept of additive calibration constant
_l = np.array([1.0,1.1,0.9]) # nsensors; intercept of multiplicative calibration constant
_sigmapsquared=np.array([0.05,0.03,0.06])**2
_kappa=np.array([[0.0,0.0,0.0],[0,0,0]])#nfac, nsensors; exponents of variance dependence
_mu=np.array([[0.0,0.00,0.0],[0,0,0]])#nfac, nsensors; slope terms of dependence of additive calibration constant
_lambda=np.array([[-0.0,0.00,0.0],[0,0,0]])#nfac, nsensors; slope terms of dependence of multiplicative calibration constant

# prepare kappa weights
explankappa=numpy_rng.uniform(low=0.1,high=1.0,size=(2,n)) # nfac, n
if explankappa is None:
    explankappa = np.zeros((0,n))
nfackappa=explankappa.shape[0]
weightkappa,normfactorkappa = normalized_weight_multiplicative(explankappa)

# prepare mu weights
if explanmu is None:
    explanmu = explankappa
nfacmu = explanmu.shape[0]
weightmu,normfactormu = normalized_weight_additive(explanmu)

# prepare lambda weights
if explanlambda is None:
    explanlambda = explankappa
nfaclambda = explanlambda.shape[0]
weightlambda,normfactorlambda = normalized_weight_additive(explanlambda)

# prepare alpha/beta weights
if explanalphabeta is None:
    explanalphabeta = explankappa
nfacalphabeta = explanlambda.shape[0]
weightalphabeta,normfactoralphabeta = normalized_weight_additive(explanalphabeta)


# simulate soil moisture (uniform for now, add options later)
theta=numpy_rng.uniform(low=0,high=0.4,size=n)

# assemble soil moisture products
# modelled variance
# model parameterization from Bayesian Computation for Parametric Models of Heteroscedasticity in the Linear Model; Boscardin and Gelman 
_sigmasquared = _sigmapsquared[:,np.newaxis]*np.prod(np.power(weightkappa[:,np.newaxis,:],_kappa[:,:,np.newaxis]),axis=0) # inside product: first dimension explanatory factor, second dimension product, third dimension time
# noise, take non-normal noise as well
noise=np.sqrt(_sigmasquared)*numpy_rng.normal(size=(nsensors,n))
# additive calibration constant M
_M=_m[:,np.newaxis]+np.sum(_mu[:,:,np.newaxis]*weightmu[:,np.newaxis,:],axis=0) # inside sum: first dimension explanatory factor, second dimension product, third dimension time
# multiplicative calibration constant L
_L=_l[:,np.newaxis]+np.sum(_lambda[:,:,np.newaxis]*weightlambda[:,np.newaxis,:],axis=0) # inside sum: first dimension explanatory factor, second dimension product, third dimension time

# product = M + L (theta - thetaoffset) + noise
y = _M+_L*(theta[np.newaxis,:]-thetaoffset)+noise

if __name__=='__main__':
    model = pm.Model()
    with model:
        
        # m and l for reference product 0
        m0=tt.zeros((1))+0.0
        l0=tt.zeros((1))+1.0
        # m and l priors for the remaining products        
        mest = pm.StudentT('mest', dof, mu=0, sd=0.3,shape=(nsensors-1))
        lest = pm.StudentT('lest', dof, mu=1, sd=0.3,shape=(nsensors-1))
        
        # define mu and M for remaining products depending on whether mu is estimated or set to zero
        if estimatesdmu:
            sdmu = pm.Exponential('sdmu',1/0.3)
        else:
            sdmu = pm.Deterministic('sdmu',tt.ones(1)*0.3)
        if estimatemu:
            muest = pm.StudentT('muest',dof,mu=0,sd=sdmu,shape=(nfacmu,nsensors-1))
            Mest = mest[:,np.newaxis] + tt.sum(muest[:,:,np.newaxis]*weightmu[:,np.newaxis,:],axis=0) #inside the sum: first dimension: explan. factor, second dimension: product, third dimension: time 
        else:
            muest = pm.Deterministic('muest',tt.zeros((nfacmu,nsensors-1))) 
            Mest = mest[:,np.newaxis]
        # same for mu0 and M0
        if estimatemu0:
            mu0est = pm.StudentT('mu0est',dof,mu=0,sd=sdmu,shape=(nfacmu))
            M0est = m0 + tt.sum(mu0est[:,np.newaxis]*weightmu,axis=0)
        else:
            mu0est = pm.Deterministic('mu0est',tt.zeros(nfacmu))
            M0est = m0
        
        # define lambda and L for remaing products depending on whether lambda is estimated or set to zero
        if estimatesdlambda:
            sdlambda = pm.Exponential('sdlambda',1/0.3)
        else:
            sdlambda = pm.Deterministic('sdlambda',tt.ones(1)*0.3)
        if estimatelambda:
            lambdaest = pm.StudentT('lambdaest',dof,mu=0,sd=sdlambda,shape=(nfaclambda,nsensors-1))
            Lest = lest[:,np.newaxis] + tt.sum(lambdaest[:,:,np.newaxis]*weightlambda[:,np.newaxis,:],axis=0)
        else:
            lambdaest = pm.Deterministic('lambdaest',tt.zeros((nfaclambda,nsensors-1)))
            Lest = lest[:,np.newaxis]
        # same for lambda0 and L0
        if estimatelambda0:
            lambda0est = pm.StudentT('lambda0est',dof,mu=0,sd=sdlambda,shape=(nfaclambda))
            L0est = l0 + tt.sum(lambda0est[:,np.newaxis]*weightlambda,axis=0)
        else:
            lambda0est = pm.Deterministic('lambda0est',tt.zeros(nfaclambda))
            L0est = l0
        
        # prior for product noise variance (all explanatory factors set to 1)
        sigmapsquared=pm.Exponential('sigmapsquared', 1/0.1, shape=(nsensors))
        # associated standard deviation for ease of reference
        sigmap=pm.Deterministic('sigmap',tt.sqrt(sigmapsquared))
        # define kappa and predicted product noise variance depending on how/whether kappa is estimated or not
        if estimatesdkappa:
            sdkappa = pm.Exponential('sdkappa',1/1.0)
        else:
            sdkappa = pm.Deterministic('sdkappa',tt.ones(1)*1.0)       
        if estimatekappa:
            if estimatekappa0:
                kappaest=pm.StudentT('kappaest',dof,mu=0,sd=sdkappa,shape=(nfackappa,nsensors)) #note that for kappa all sensors (including reference sensor) are represented in the same variable
                kappa=pm.Deterministic('kappa',1.0*kappaest)
            else:
                kappaest=pm.StudentT('kappaest',dof,mu=0,sd=sdkappa,shape=(nfackappa,nsensors-1))
                kappa0=tt.zeros((nfackappa,1))
                kappa=pm.Deterministic('kappa',tt.concatenate([kappa0,kappaest],axis=1))
            sigmasquaredtotal=(sigmapsquared[:,np.newaxis]*tt.prod(tt.pow(weightkappa[:,np.newaxis,:],kappa[:,:,np.newaxis]),axis=0))
        else:
            kappa = pm.Deterministic('kappa',tt.zeros((1,nsensors)))
            sigmasquaredtotal=sigmapsquared[:,np.newaxis]
        
        # porosity, i.e. maximum soil moisture content: T prior
        porosity = pm.StudentT('porosity',dof,mu=0.4,sd=0.1)
        
        # distribution of theta
        if thetamodel == 'beta':
            # beta distribution (A=alpha and B=beta estimated), can vary with explan. factors
            a = pm.ChiSquared('a',dofchi)
            b = pm.ChiSquared('b',dofchi)
            if estimatesdalphabeta:
                sdalphabeta = pm.Exponential('sdalphabeta',1/0.3)
            else:
                sdalphabeta = pm.Deterministic('sdalphabeta',tt.ones(1)*0.3)            
            if estimatealphabeta:
                alpha=pm.StudentT('alpha',dof,mu=0,sd=sdalphabeta,shape=(nfacalphabeta))
                beta=pm.StudentT('beta',dof,mu=0,sd=sdalphabeta,shape=(nfacalphabeta))            
                A = tt.sqrt(softabsvalue**2+tt.pow(a + tt.sum(alpha[:,np.newaxis]*weightalphabeta,axis=0)),2) # soft absolute value; A and B should be >> softabsvalue
                B = tt.sqrt(softabsvalue**2+tt.pow(b + tt.sum(beta[:,np.newaxis]*weightalphabeta,axis=0)),2)
            else:
                alpha = pm.Deterministic('alpha',tt.zeros(1)*0.0)
                beta = pm.Deterministic('beta', tt.zeros(1)*0.0)
                A = a
                B = b
            thetaub = pm.Beta('thetaub',alpha=A,beta=B, shape=(n))
            theta = pm.Deterministic('theta',porosity*thetaub)
        elif thetamodel == 'logistic':
            # spline with logistic link
            a = pm.StudentT('a',dof,mu=0.0,sd=3.0)
            b = pm.Exponential('b',1./3)
            if estimatesdalphabeta:
                sdalphabeta = pm.Exponential('sdalphabeta',1.0)
            else:
                sdalphabeta = pm.Deterministic('sdalphabeta',tt.ones(1)*1.0)              
            if estimatealphabeta:
                alpha=pm.StudentT('alpha',dof,mu=0,sd=sdalphabeta,shape=(nfacalphabeta))
                beta=pm.StudentT('beta',dof,mu=0,sd=sdalphabeta,shape=(nfacalphabeta))            
                A = a + tt.sum(alpha[:,np.newaxis]*weightalphabeta,axis=0)
                B = tt.sqrt(softabsvalue**2+tt.pow(b + tt.sum(beta[:,np.newaxis]*weightalphabeta,axis=0)),2)
            else:
                alpha = pm.Deterministic('alpha',tt.zeros(1)*0.0)
                beta = pm.Deterministic('beta', tt.zeros(1)*0.0)
                A = a
                B = b      
            thetaub = pm.Normal('thetaub',mu=A, sd=B, shape=(n)) 
            theta = pm.Deterministic('theta',porosity*tt.pow(1+tt.exp(-thetaub), -1)) 
        # assemble mean of observed products
        y0=M0est+L0est*(theta[np.newaxis,:]-thetaoffset)
        yrest = Mest+Lest*(theta[np.newaxis,:]-thetaoffset) 
        yest=tt.concatenate([y0,yrest],axis=0)
        
        # model for observed products
        y = pm.Normal('y', mu=yest, sd=tt.sqrt(sigmasquaredtotal), observed=y)#sigma[:,np.newaxis]      
    print(model.unobserved_RVs)
    with model:
        # inference
        v_params = pm.variational.advi(n=200000, random_seed=seed)
        tracevi = pm.variational.sample_vp(v_params, draws=1000, random_seed=seed)
        step = pm.NUTS(scaling=np.power(model.dict_to_array(v_params.stds), 2), is_cov=True)
        trace = pm.sample(draws=niter, step=step, start=v_params.means,random_seed=seed)        
       
        #pm.summary(tracevi,varnames=['sigmap','mest','lest','porosity','a','b','kappa'])
        print('-------------')
        pm.summary(trace[niter//2::],varnames=['sigmap','mest','lest','sdmu','sdlambda','sdkappa','porosity','a','b','kappa','mu0est','muest','lambda0est','lambdaest'])
        
        plt.show()
