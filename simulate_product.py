'''
Created on Jun 1, 2017

@author: zwieback
'''
import numpy as np
from normalize_weights import normalize_weights

def simulate_product(n, params, numpy_rng=None, seed=123):
    # n: number of observations
    # numpy_rng: numpy random number generator
    # seed: seed to initialize numpy_rng with if numpy_rng is not provided   
    
    if numpy_rng is None:
        # initialize random number generator
        numpy_rng = np.random.RandomState(seed)
    
    paramssim = {}
    
    paramssim['sigmapsquared'] = params['sigmapsquared'] if 'sigmapsquared' in params else np.array([0.02,0.02,0.02])**2 # nsensors; measurement error variance
    paramssim['l'] = params['l'] if 'l' in params else np.array([1,1,1]) # nsensors; intercept of multiplicative calibration constant
    paramssim['m'] = params['m'] if 'm' in params else np.array([0,0,0]) # nsenors; intercept of additive calibration constant
    nsensors = len(paramssim['sigmapsquared'])

    paramssim['soilmoisturemodel'] = params['soilmoisturemodel'] if 'soilmoisturemodel' in params else 'uniform' 
    paramssim['porosity'] = params['porosity'] if 'porosity' in params else 0.4
    paramssim['thetaoffset'] = params['thetaoffset'] if 'thetaoffset' in params else 0.15

    # simulate soil moisture
    if paramssim['soilmoisturemodel'] == 'uniform':
        theta = numpy_rng.uniform(low = 0, high = paramssim['porosity'], size = n)
        fy = numpy_rng.uniform(size = (n))
        fy = np.sort(fy) # only one year 
        day = (fy*365.25).astype(np.int)
    elif paramssim['soilmoisturemodel'] == 'smapapi':
        from simulate_soil_moisture_smap import soil_moisture_smap_time_series
        import math
        paramssim['prob_dry_model'] = params['prob_dry_model'] if 'prob_dry_model' in params else lambda fy: np.array([0.55-0.1*math.cos(2*math.pi*fy), 0.85-0.1*math.cos(2*math.pi*fy)])
        paramssim['mean_rainfall_model'] = params['mean_rainfall_model'] if 'mean_rainfall_model' in params else lambda fy: 0.006
        paramssim['loss_model'] = params['loss_model'] if 'loss_model' in params else lambda fy: 0.9+0.06*math.cos(2*math.pi*fy)
        day,fy,theta = soil_moisture_smap_time_series(numpy_rng, n, paramssim['prob_dry_model'], paramssim['mean_rainfall_model'], paramssim['loss_model'], porosity = paramssim['porosity'])
    else:
        raise NotImplementedError    
    
    
    paramssim['spline'] = params['spline'] if 'spline' in params else False
    explan={}
    if paramssim['spline']:
        from spline_predictors import periodic_spline_predictors        
        nsegments = params['nsegments_spline'] if 'nsegments_spline' in params else 12
        psp = periodic_spline_predictors(fy,nsegments=nsegments)
        paramssim['explankappa'] = paramssim['explanmu'] = paramssim['explanlambda'] = psp['basisfunctions']['diffcoeff']
        paramssim['nfackappa'] = paramssim['nfacmu'] = paramssim['nfaclambda'] = nsegments
        for paramname in ['kappa','mu','lambda']:
            paramssim[paramname] = params[paramname] if paramname in params else np.zeros((nsegments,nsensors))
    else:    
        for paramname in ['kappa','mu','lambda']: #exponents of variance dependence; slope terms of dependence of additive calibration constant; slope terms of dependence of multiplicative calibration constant
            paramssim['nfac'+paramname] = params['explan'+paramname].shape[0] if 'explan'+paramname in params else 0
            paramssim['explan'+paramname] = params['explan'+paramname] if 'explan'+paramname in params else None    
            paramssim[paramname] = params[paramname] if paramname in params else np.zeros((paramssim['nfac'+paramname],nsensors))
    
    for paramname in ['kappa','mu','lambda']:
        if 'explanmodel'+paramname in params: # if model exists, overwrite previous settings
            paramssim['explan'+paramname] = params['explanmodel'+paramname](fy)
            paramssim['nfac'+paramname] = params['explan'+paramname].shape[0] if 'explan'+paramname in params else 0
        explan[paramname] = paramssim['explan'+paramname]
    
    normalized_weights = normalize_weights(explankappa=paramssim['explankappa'], explanmu=paramssim['explanmu'], explanlambda=paramssim['explanlambda'], n=n)
    
    # assemble soil moisture products
    # modelled variance
    # model parameterization from Bayesian Computation for Parametric Models of Heteroscedasticity in the Linear Model; Boscardin and Gelman 
    _sigmasquared = paramssim['sigmapsquared'][:,np.newaxis]*np.prod(np.power(normalized_weights['kappa']['weight'][:,np.newaxis,:],paramssim['kappa'][:,:,np.newaxis]),axis=0) # inside product: first dimension explanatory factor, second dimension product, third dimension time
    # noise, take non-normal noise as well + AR(1) with rho = 0.5: mixing with 1/2 (previous final noise term) and sqrt(3)/2 (current noise term before mixing)
    paramssim['noisedistribution'] = params['noisedistribution'] if 'noisedistribution' in params else 'normal'
    if paramssim['noisedistribution'] ==  'normal':
        noise = np.sqrt(_sigmasquared)*numpy_rng.normal(size=(nsensors,n))
    elif paramssim['noisedistribution'] == 'studentt':
        paramssim['noise_dof'] = params['noise_dof'] if 'noise_dof' in params else 6
        noise = np.sqrt(_sigmasquared)*np.sqrt((paramssim['noise_dof']-2)/paramssim['noise_dof'])*numpy_rng.standard_t(paramssim['noise_dof'],size=(nsensors,n)) # var(standard_t) != 1 
    elif paramssim['noisedistribution'] == 'normal_ar1': # this is only an approximation if _sigmasquared varies over time
        noisewhite = np.sqrt(_sigmasquared)*numpy_rng.normal(size=(nsensors,n))
        noiseinnov = np.sqrt(_sigmasquared)*numpy_rng.normal(size=(nsensors,n))                
        weightprev = 0.5
        weightinnov = np.sqrt(3)/2
        noise = noisewhite.copy()
        for j in np.arange(1, n):
            noise[:,j] = weightprev*noise[:,j-1] + weightinnov*noiseinnov[:,j-1]
    else:
        raise NotImplementedError
    # additive calibration constant M
    _M = paramssim['m'][:,np.newaxis]+np.sum(paramssim['mu'][:,:,np.newaxis]*normalized_weights['mu']['weight'][:,np.newaxis,:],axis=0) # inside sum: first dimension explanatory factor, second dimension product, third dimension time
    # multiplicative calibration constant L
    _L = paramssim['l'][:,np.newaxis]+np.sum(paramssim['lambda'][:,:,np.newaxis]*normalized_weights['lambda']['weight'][:,np.newaxis,:],axis=0) # inside sum: first dimension explanatory factor, second dimension product, third dimension time
    # product = M + L (theta - thetaoffset) + noise
    y = _M + _L*(theta[np.newaxis,:]-paramssim['thetaoffset']) + paramssim['thetaoffset'] + noise
    
    visible = {'y':y, 'explan':explan, 'fy':fy, 'day':day}
    internal = {'theta':theta, 'noise':noise, 'params': paramssim}
    
    return visible,internal,normalized_weights
    