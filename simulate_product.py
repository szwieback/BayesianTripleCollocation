'''
Created on Jun 1, 2017

@author: zwieback
'''
import numpy as np
from normalize_weights import normalize_weights


def simulate_product(nsensors,n,porosity=0.4,thetaoffset=0.2,numpy_rng=None,seed=123):
    # n: number of observations
    # nsensors: number of sensors/products
    # numpy_rng: numpy random number generator
    # seed: seed to initialize numpy_rng with if numpy_rng is not provided
    
    # soft-code most specifications: input dictionary?
    
    if numpy_rng is None:
        # initialize random number generator
        numpy_rng = np.random.RandomState(seed)
    
    _m = np.array([0.0,0.1,-0.15]) #nsensors;  intercept of additive calibration constant
    _l = np.array([1.0,1.1,0.9]) # nsensors; intercept of multiplicative calibration constant
    _sigmapsquared=np.array([0.05,0.03,0.06])**2
    _kappa=np.array([[0.0,0.0,0.0],[0,0,0]])#nfac, nsensors; exponents of variance dependence
    _mu=np.array([[0.0,0.00,0.0],[0,0,0]])#nfac, nsensors; slope terms of dependence of additive calibration constant
    _lambda=np.array([[-0.0,0.00,0.0],[0,0,0]])#nfac, nsensors; slope terms of dependence of multiplicative calibration constant
    explan=numpy_rng.uniform(low=0.1,high=1.0,size=(2,n)) # nfac, n
    normalized_weights=normalize_weights(explankappa=explan, explanmu=explan, explanlambda=explan, explanalphabeta=explan, n=n)
    # simulate soil moisture (uniform for now, add options later)
    theta=numpy_rng.uniform(low=0,high=porosity,size=n)
    # assemble soil moisture products
    # modelled variance
    # model parameterization from Bayesian Computation for Parametric Models of Heteroscedasticity in the Linear Model; Boscardin and Gelman 
    _sigmasquared = _sigmapsquared[:,np.newaxis]*np.prod(np.power(normalized_weights['kappa']['weight'][:,np.newaxis,:],_kappa[:,:,np.newaxis]),axis=0) # inside product: first dimension explanatory factor, second dimension product, third dimension time
    # noise, take non-normal noise as well
    noise=np.sqrt(_sigmasquared)*numpy_rng.normal(size=(nsensors,n))
    # additive calibration constant M
    _M=_m[:,np.newaxis]+np.sum(_mu[:,:,np.newaxis]*normalized_weights['mu']['weight'][:,np.newaxis,:],axis=0) # inside sum: first dimension explanatory factor, second dimension product, third dimension time
    # multiplicative calibration constant L
    _L=_l[:,np.newaxis]+np.sum(_lambda[:,:,np.newaxis]*normalized_weights['lambda']['weight'][:,np.newaxis,:],axis=0) # inside sum: first dimension explanatory factor, second dimension product, third dimension time
    # product = M + L (theta - thetaoffset) + noise
    y = _M+_L*(theta[np.newaxis,:]-thetaoffset)+noise
    
    visible = {'y':y, 'explan':explan}
    internal = {'theta':theta,'noise':noise}
    
    return visible,internal,normalized_weights
    