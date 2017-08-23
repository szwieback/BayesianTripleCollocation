'''
Created on Jun 1, 2017

@author: zwieback
'''
import pymc3 as pm
import theano.tensor as tt
import numpy as np
def model_setup(visible,normalized_weights,estimateexplanterms={},estimatesdexplanterms={},inferenceparams={}):
    nsensors=visible['y'].shape[0]
    n=visible['y'].shape[1]
    
    model = pm.Model()
    with model:
        doft = inferenceparams['doft'] if 'doft' in inferenceparams else 4
        priorfactor = inferenceparams['priorfactor'] if 'priorfactor' in inferenceparams else 1.0
        
        # m and l for reference product 0
        m0=tt.zeros((1))+0.0
        l0=tt.zeros((1))+1.0
        # m and l priors for the remaining products        
        mest = pm.StudentT('mest', doft, mu=0, sd=0.3*priorfactor,shape=(nsensors-1))
        lest = pm.StudentT('lest', doft, mu=1, sd=0.3*priorfactor,shape=(nsensors-1))
        
        # define mu and M for remaining products depending on whether mu is estimated or set to zero
        if 'mu' in estimatesdexplanterms and estimatesdexplanterms['mu'] and normalized_weights['mu']['weight'].shape[0]>0:
            sdmu = pm.Exponential('sdmu', 1/(0.3*priorfactor))
        else:
            sdmu = pm.Deterministic('sdmu', tt.ones(1)*0.3*priorfactor)
        if 'mu' in estimateexplanterms and estimateexplanterms['mu'] and normalized_weights['mu']['weight'].shape[0]>0:            
            muestnondim = pm.StudentT('muestnondim', doft, mu=0, sd=1, shape=(normalized_weights['mu']['nfac'], nsensors-1))
            muest = pm.Deterministic('muest', muestnondim*sdmu)
            Mest = mest[:,np.newaxis] + tt.sum(muest[:,:,np.newaxis]*normalized_weights['mu']['weight'][:,np.newaxis,:], axis=0) #inside the sum: first dimension: explan. factor, second dimension: product, third dimension: time 
        else:
            muest = pm.Deterministic('muest',tt.zeros((normalized_weights['mu']['nfac'], nsensors-1))) 
            Mest = mest[:,np.newaxis]
        # same for mu0 and M0
        if 'mu0' in estimateexplanterms and estimateexplanterms['mu0'] and normalized_weights['mu']['weight'].shape[0]>0:
            mu0estnondim = pm.StudentT('mu0estnondim', doft, mu=0, sd=1, shape=(normalized_weights['mu']['nfac']))
            mu0est = pm.Deterministic('mu0est',mu0estnondim*sdmu)
            M0est = m0 + tt.sum(mu0est[:,np.newaxis]*normalized_weights['mu']['weight'], axis=0)
        else:
            mu0est = pm.Deterministic('mu0est',tt.zeros(normalized_weights['mu']['nfac']))
            M0est = m0
        
        # define lambda and L for remaining products depending on whether lambda is estimated or set to zero
        if 'lambda' in estimatesdexplanterms and estimatesdexplanterms['lambda'] and normalized_weights['lambda']['weight'].shape[0]>0:
            sdlambda = pm.Exponential('sdlambda', 1/(0.3*priorfactor))
        else:
            sdlambda = pm.Deterministic('sdlambda', tt.ones(1)*0.3*priorfactor)
        if 'lambda' in estimateexplanterms and estimateexplanterms['lambda'] and normalized_weights['lambda']['weight'].shape[0]>0:
            lambdaestnondim = pm.StudentT('lambdaestnondim', doft, mu=0, sd=1, shape=(normalized_weights['lambda']['nfac'], nsensors-1))
            lambdaest = pm.Deterministic('lambdaest', lambdaestnondim*sdlambda)            
            Lest = lest[:,np.newaxis] + tt.sum(lambdaest[:,:,np.newaxis]*normalized_weights['lambda']['weight'][:,np.newaxis,:], axis=0)
        else:
            lambdaest = pm.Deterministic('lambdaest', tt.zeros((normalized_weights['lambda']['nfac'],nsensors-1)))
            Lest = lest[:,np.newaxis]
        # same for lambda0 and L0
        if 'lambda0' in estimateexplanterms and estimateexplanterms['lambda0'] and normalized_weights['lambda']['weight'].shape[0]>0:
            lambda0estnondim = pm.StudentT('lambda0estnondim', doft, mu=0, sd=1, shape=(normalized_weights['lambda']['nfac']))
            lambda0est=pm.Deterministic('lambda0est',lambda0estnondim*sdlambda)
            L0est = l0 + tt.sum(lambda0est[:,np.newaxis]*normalized_weights['lambda']['weight'], axis=0)
        else:
            lambda0est = pm.Deterministic('lambda0est', tt.zeros(normalized_weights['lambda']['nfac']))
            L0est = l0
        
        # product variance        
        if 'sigmap0' in estimateexplanterms and not estimateexplanterms['sigmap0']:
            # set sigmap0 to fixed value
            sigmap0value = inferenceparams['sigmap0'] if 'sigmap0' in inferenceparams else 0.01
            sigmap0squared = pm.Deterministic('sigmap0squared', tt.ones(1)*sigmap0value**2)
            sigmapsquaredest = pm.Exponential('sigmapsquaredest', 1/(0.1*priorfactor), shape=(nsensors-1))
            sigmapsquared = tt.concatenate([sigmap0squared,sigmapsquaredest], axis=0)
        else:
            # prior for product noise variance (all explanatory factors set to 1)
            sigmapsquared=pm.Exponential('sigmapsquared', 1/(0.1*priorfactor), shape=(nsensors))
        # associated standard deviation for ease of reference
        sigmap=pm.Deterministic('sigmap', tt.sqrt(sigmapsquared))
        # define kappa and predicted product noise variance depending on how/whether kappa is estimated or not
        if 'kappa' in estimatesdexplanterms and estimatesdexplanterms['kappa'] and normalized_weights['kappa']['weight'].shape[0]>0:
            sdkappa = pm.Exponential('sdkappa', 1/(1.0*priorfactor))
        else:
            sdkappa = pm.Deterministic('sdkappa', tt.ones(1)*1.0*priorfactor)       
        if 'kappa' in estimateexplanterms and estimateexplanterms['kappa'] and normalized_weights['kappa']['weight'].shape[0]>0:
            if 'kappa0' in estimateexplanterms and estimateexplanterms['kappa0']:
                kappaestnondim = pm.StudentT('kappaestnondim', doft, mu=0, sd=1, shape=(normalized_weights['kappa']['nfac'], nsensors)) #note that for kappa all sensors (including reference sensor) are represented in the same variable
                kappaest = pm.Deterministic('kappaest', kappaestnondim*sdkappa)
                kappa = pm.Deterministic('kappa', 1.0*kappaest)
            else:
                kappaestnondim = pm.StudentT('kappaestnondim', doft, mu=0, sd=1, shape=(normalized_weights['kappa']['nfac'], nsensors-1))
                kappaest = pm.Deterministic('kappaest', kappaestnondim*sdkappa)
                kappa0 = tt.zeros((normalized_weights['kappa']['nfac'], 1))
                kappa = pm.Deterministic('kappa', tt.concatenate([kappa0,kappaest], axis=1))
            sigmasquaredtotal = (sigmapsquared[:,np.newaxis]*tt.prod(tt.pow(normalized_weights['kappa']['weight'][:,np.newaxis,:], kappa[:,:,np.newaxis]), axis=0))
        else:
            kappa = pm.Deterministic('kappa',tt.zeros((1,nsensors)))
            sigmasquaredtotal = sigmapsquared[:,np.newaxis]*tt.ones((nsensors,n))
        
        # porosity, i.e. maximum soil moisture content: T prior
        porosity = pm.StudentT('porosity',doft,mu=0.4,sd=0.1*priorfactor)
        
        thetamodel = inferenceparams['thetamodel'] if 'thetamodel' in inferenceparams else 'beta'
        dofchi = inferenceparams['dofchi'] if 'dofchi' in inferenceparams else 3
        softabsvalue = inferenceparams['softabsvalue'] if 'softabsvalue' in inferenceparams else 0.01
        # distribution of theta
        if thetamodel == 'beta':
            # beta distribution (A=alpha and B=beta estimated), can vary with explan. factors
            a = pm.ChiSquared('a', dofchi)
            b = pm.ChiSquared('b', dofchi)
            if 'sdalphabeta' in estimatesdexplanterms and estimatesdexplanterms['alphabeta']:
                sdalphabeta = pm.Exponential('sdalphabeta', 1/(0.3*priorfactor))
            else:
                sdalphabeta = pm.Deterministic('sdalphabeta', tt.ones(1)*0.3*priorfactor)            
            if 'alphabeta' in estimateexplanterms and estimateexplanterms['alphabeta']:
                alphanondim = pm.StudentT('alphanondim', doft, mu=0, sd=1,shape=(normalized_weights['alphabeta']['nfac']))
                betanondim = pm.StudentT('betanondim', doft, mu=0, sd=1,shape=(normalized_weights['alphabeta']['nfac']))
                alpha = pm.Deterministic('alpha', alphanondim*sdalphabeta)
                beta = pm.Deterministic('beta', betanondim*sdalphabeta)                                     
                A = tt.sqrt(softabsvalue**2+tt.pow(a + tt.sum(alpha[:,np.newaxis]*normalized_weights['alphabeta']['weight'],axis=0),2)) # soft absolute value; A and B should be >> softabsvalue
                B = tt.sqrt(softabsvalue**2+tt.pow(b + tt.sum(beta[:,np.newaxis]*normalized_weights['alphabeta']['weight'],axis=0),2))
            else:
                alpha = pm.Deterministic('alpha', tt.zeros(1)*0.0)
                beta = pm.Deterministic('beta', tt.zeros(1)*0.0)
                A = a
                B = b
            thetaub = pm.Beta('thetaub', alpha=A, beta=B, shape=(n))
            theta = pm.Deterministic('theta', porosity*thetaub)
        elif thetamodel == 'logistic':
            # spline with logistic link
            a = pm.StudentT('a', doft, mu=0.0, sd=3.0*priorfactor)
            b = pm.Exponential('b', 1./(3*priorfactor))
            if 'sdalphabeta' in estimatesdexplanterms and estimatesdexplanterms['alphabeta']:
                sdalphabeta = pm.Exponential('sdalphabeta', 1.0/(1.0*priorfactor))
            else:
                sdalphabeta = pm.Deterministic('sdalphabeta',tt.ones(1)*1.0*priorfactor)              
            if 'alphabeta' in estimateexplanterms and estimateexplanterms['alphabeta']:
                alphanondim = pm.StudentT('alphanondim', doft, mu=0,sd=1, shape=(normalized_weights['alphabeta']['nfac']))
                betanondim  = pm.StudentT('betanondim', doft, mu=0, sd=1, shape=(normalized_weights['alphabeta']['nfac']))
                alpha = pm.Deterministic('alpha', alphanondim*sdalphabeta)
                beta = pm.Deterministic('beta', betanondim*sdalphabeta)             
                A = a + tt.sum(alpha[:,np.newaxis]*normalized_weights['alphabeta']['weight'], axis=0)
                B = tt.sqrt(softabsvalue**2+tt.pow(b + tt.sum(beta[:,np.newaxis]*normalized_weights['alphabeta']['weight'],axis=0),2))
            else:
                alpha = pm.Deterministic('alpha',tt.zeros(1)*0.0)
                beta = pm.Deterministic('beta', tt.zeros(1)*0.0)
                A = a
                B = b      
            thetaubnondim = pm.Normal('thetaubnondim', mu=0, sd=1, shape=(n))
            thetaub = pm.Deterministic('thetaub', A+B*thetaubnondim)
            theta = pm.Deterministic('theta', porosity*tt.pow(1+tt.exp(-thetaub), -1)) 
        
        # assemble mean of observed products
        thetaoffset = inferenceparams['thetaoffset'] if 'thetaoffset' in inferenceparams else 0.15
        if thetaoffset == 'observedmean':
            thetaoffset = np.mean(visible['y'][0,:])        
        y0 = M0est+L0est*(theta[np.newaxis,:] - thetaoffset)
        yrest = Mest+Lest*(theta[np.newaxis,:] - thetaoffset) 
        yest = tt.concatenate([y0,yrest], axis=0) + thetaoffset
        
        # model for observed products
        studenterrors = inferenceparams['studenterrors'] if 'studenterrors' in inferenceparams else False
        if not studenterrors:
            y = pm.Normal('y', mu=yest, sd=tt.sqrt(sigmasquaredtotal), observed=visible['y'])
        else:
            studenterrors_dof = inferenceparams['studenterrors_dof'] if 'studenterrors_dof' in inferenceparams else doft
            lam = tt.pow(sigmasquaredtotal, -1)*(studenterrors_dof/(studenterrors_dof-2))
            y = pm.StudentT('y', studenterrors_dof, mu=yest, lam=lam, observed=visible['y'])
    return model

