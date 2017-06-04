'''
Created on Jun 1, 2017

@author: zwieback
'''
import pymc3 as pm
import theano.tensor as tt
import numpy as np
def model_setup(visible,normalized_weights,estimateexplanterms={},estimatesdexplanterms={},thetamodel='beta',thetaoffset=0.15,doft=4,dofchi=3,priorfactor=1.0,softabsvalue=0.01):
    nsensors=visible['y'].shape[0]
    n=visible['y'].shape[1]
    
    model = pm.Model()
    with model:
        
        # m and l for reference product 0
        m0=tt.zeros((1))+0.0
        l0=tt.zeros((1))+1.0
        # m and l priors for the remaining products        
        mest = pm.StudentT('mest', doft, mu=0, sd=0.3*priorfactor,shape=(nsensors-1))
        lest = pm.StudentT('lest', doft, mu=1, sd=0.3*priorfactor,shape=(nsensors-1))
        
        # define mu and M for remaining products depending on whether mu is estimated or set to zero
        if 'mu' in estimatesdexplanterms and estimatesdexplanterms['mu']:
            sdmu = pm.Exponential('sdmu',1/(0.3*priorfactor))
        else:
            sdmu = pm.Deterministic('sdmu',tt.ones(1)*0.3*priorfactor)
        if 'mu' in estimateexplanterms and estimateexplanterms['mu']:
            muest = pm.StudentT('muest',doft,mu=0,sd=sdmu,shape=(normalized_weights['mu']['nfac'],nsensors-1))
            Mest = mest[:,np.newaxis] + tt.sum(muest[:,:,np.newaxis]*normalized_weights['mu']['weight'][:,np.newaxis,:],axis=0) #inside the sum: first dimension: explan. factor, second dimension: product, third dimension: time 
        else:
            muest = pm.Deterministic('muest',tt.zeros((normalized_weights['mu']['nfac'],nsensors-1))) 
            Mest = mest[:,np.newaxis]
        # same for mu0 and M0
        if 'mu0' in estimateexplanterms and estimateexplanterms['mu0']:
            mu0est = pm.StudentT('mu0est',doft,mu=0,sd=sdmu,shape=(normalized_weights['mu']['nfac']))
            M0est = m0 + tt.sum(mu0est[:,np.newaxis]*normalized_weights['mu']['weight'],axis=0)
        else:
            mu0est = pm.Deterministic('mu0est',tt.zeros(normalized_weights['mu']['nfac']))
            M0est = m0
        
        # define lambda and L for remaining products depending on whether lambda is estimated or set to zero
        if 'lambda' in estimatesdexplanterms and estimatesdexplanterms['lambda']:
            sdlambda = pm.Exponential('sdlambda',1/(0.3*priorfactor))
        else:
            sdlambda = pm.Deterministic('sdlambda',tt.ones(1)*0.3*priorfactor)
        if 'lambda' in estimateexplanterms and estimateexplanterms['lambda']:
            lambdaest = pm.StudentT('lambdaest',doft,mu=0,sd=sdlambda,shape=(normalized_weights['lambda']['nfac'],nsensors-1))
            Lest = lest[:,np.newaxis] + tt.sum(lambdaest[:,:,np.newaxis]*normalized_weights['lambda']['weight'][:,np.newaxis,:],axis=0)
        else:
            lambdaest = pm.Deterministic('lambdaest',tt.zeros((normalized_weights['lambda']['nfac'],nsensors-1)))
            Lest = lest[:,np.newaxis]
        # same for lambda0 and L0
        if 'lambda0' in estimateexplanterms and estimateexplanterms['lambda0']:
            lambda0est = pm.StudentT('lambda0est',doft,mu=0,sd=sdlambda,shape=(normalized_weights['lambda']['nfac']))
            L0est = l0 + tt.sum(lambda0est[:,np.newaxis]*normalized_weights['lambda']['weight'],axis=0)
        else:
            lambda0est = pm.Deterministic('lambda0est',tt.zeros(normalized_weights['lambda']['nfac']))
            L0est = l0
        
        # prior for product noise variance (all explanatory factors set to 1)
        sigmapsquared=pm.Exponential('sigmapsquared', 1/(0.1*priorfactor), shape=(nsensors))
        # associated standard deviation for ease of reference
        sigmap=pm.Deterministic('sigmap',tt.sqrt(sigmapsquared))
        # define kappa and predicted product noise variance depending on how/whether kappa is estimated or not
        if 'kappa' in estimatesdexplanterms and estimatesdexplanterms['kappa']:
            sdkappa = pm.Exponential('sdkappa',1/(1.0*priorfactor))
        else:
            sdkappa = pm.Deterministic('sdkappa',tt.ones(1)*1.0*priorfactor)       
        if 'kappa' in estimateexplanterms and estimateexplanterms['kappa']:
            if 'kappa0' in estimateexplanterms and estimateexplanterms['kappa0']:
                kappaest=pm.StudentT('kappaest',doft,mu=0,sd=sdkappa,shape=(normalized_weights['kappa']['nfac'],nsensors)) #note that for kappa all sensors (including reference sensor) are represented in the same variable
                kappa=pm.Deterministic('kappa',1.0*kappaest)
            else:
                kappaest=pm.StudentT('kappaest',doft,mu=0,sd=sdkappa,shape=(normalized_weights['kappa']['nfac'],nsensors-1))
                kappa0=tt.zeros((normalized_weights['kappa']['nfac'],1))
                kappa=pm.Deterministic('kappa',tt.concatenate([kappa0,kappaest],axis=1))
            sigmasquaredtotal=(sigmapsquared[:,np.newaxis]*tt.prod(tt.pow(normalized_weights['kappa']['weight'][:,np.newaxis,:],kappa[:,:,np.newaxis]),axis=0))
        else:
            kappa = pm.Deterministic('kappa',tt.zeros((1,nsensors)))
            sigmasquaredtotal=sigmapsquared[:,np.newaxis]
        
        # porosity, i.e. maximum soil moisture content: T prior
        porosity = pm.StudentT('porosity',doft,mu=0.4,sd=0.1*priorfactor)
        
        # distribution of theta
        if thetamodel == 'beta':
            # beta distribution (A=alpha and B=beta estimated), can vary with explan. factors
            a = pm.ChiSquared('a',dofchi)
            b = pm.ChiSquared('b',dofchi)
            if 'sdalphabeta' in estimatesdexplanterms and estimatesdexplanterms['alphabeta']:
                sdalphabeta = pm.Exponential('sdalphabeta',1/(0.3*priorfactor))
            else:
                sdalphabeta = pm.Deterministic('sdalphabeta',tt.ones(1)*0.3*priorfactor)            
            if 'alphabeta' in estimateexplanterms and estimateexplanterms['alphabeta']:
                alpha=pm.StudentT('alpha',doft,mu=0,sd=sdalphabeta,shape=(normalized_weights['alphabeta']['nfac']))
                beta=pm.StudentT('beta',doft,mu=0,sd=sdalphabeta,shape=(normalized_weights['alphabeta']['nfac']))            
                A = tt.sqrt(softabsvalue**2+tt.pow(a + tt.sum(alpha[:,np.newaxis]*normalized_weights['alphabeta']['weight'],axis=0),2)) # soft absolute value; A and B should be >> softabsvalue
                B = tt.sqrt(softabsvalue**2+tt.pow(b + tt.sum(beta[:,np.newaxis]*normalized_weights['alphabeta']['weight'],axis=0),2))
            else:
                alpha = pm.Deterministic('alpha',tt.zeros(1)*0.0)
                beta = pm.Deterministic('beta', tt.zeros(1)*0.0)
                A = a
                B = b
            thetaub = pm.Beta('thetaub',alpha=A,beta=B, shape=(n))
            theta = pm.Deterministic('theta',porosity*thetaub)
        elif thetamodel == 'logistic':
            # spline with logistic link
            a = pm.StudentT('a',doft,mu=0.0,sd=3.0*priorfactor)
            b = pm.Exponential('b',1./(3*priorfactor))
            if 'sdalphabeta' in estimatesdexplanterms and estimatesdexplanterms['alphabeta']:
                sdalphabeta = pm.Exponential('sdalphabeta',1.0/(1.0*priorfactor))
            else:
                sdalphabeta = pm.Deterministic('sdalphabeta',tt.ones(1)*1.0*priorfactor)              
            if 'alphabeta' in estimateexplanterms and estimateexplanterms['alphabeta']:
                alpha=pm.StudentT('alpha',doft,mu=0,sd=sdalphabeta,shape=(normalized_weights['alphabeta']['nfac']))
                beta=pm.StudentT('beta',doft,mu=0,sd=sdalphabeta,shape=(normalized_weights['alphabeta']['nfac']))            
                A = a + tt.sum(alpha[:,np.newaxis]*normalized_weights['alphabeta']['weight'],axis=0)
                B = tt.sqrt(softabsvalue**2+tt.pow(b + tt.sum(beta[:,np.newaxis]*normalized_weights['alphabeta']['weight'],axis=0),2))
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
        y = pm.Normal('y', mu=yest, sd=tt.sqrt(sigmasquaredtotal), observed=visible['y'])    
    return model

