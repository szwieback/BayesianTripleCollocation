'''
Created on Jun 7, 2017

@author: zwieback
'''
from copy import deepcopy
from spline_predictors import periodic_spline_predictors        
from normalize_weights import normalize_weights

def inference_definition(scenarioname):
    if scenarioname == 'Q1base':
        estimateexplanterms = {'kappa0':False, 'kappa':False, 'mu':False, 'mu0':False, 'lambda':False, 'lambda0':False, 'alphabeta':False}
        estimatesdexplanterms = {'mu':False, 'lambda':False, 'kappa':False, 'alphabeta':False}    
        inferenceparams={'thetaoffset':0.15,'thetamodel':'logistic', 'doft':4, 'dofchi':3, 'priorfactor':1.0, 'softabsvalue':0.01, 'studenterrors': False}
    elif scenarioname == 'Q1lambdamu':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1base')
        estimateexplanterms['mu'] = True
        estimateexplanterms['lambda'] = True
    elif scenarioname == 'Q1kappa':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1lambdamu')
        estimateexplanterms['kappa'] = True
    elif scenarioname in ['Q1spline', 'Q2spline_base']:
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1lambdamu')
        estimatesdexplanterms['mu'] = estimatesdexplanterms['lambda'] = True
    elif scenarioname == 'Q2kappa_lambdamu':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1lambdamu')
    elif scenarioname in ['Q2kappa_base', 'Q2lambdamu_base']:
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1base')
    #'Q3dof', 'Q3priorfactor', 'Q3beta'
    elif scenarioname =='Q3dof':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1kappa')
        inferenceparams['doft'] = 10
    elif scenarioname =='Q3priorfactor':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1kappa')
        inferenceparams['priorfactor'] = 2.0
    elif scenarioname == 'Q3beta':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1kappa')
        inferenceparams['thetamodel'] = 'beta'
    elif scenarioname == 'Q3logisticspline':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1kappa')
        estimateexplanterms['alphabeta'] = True
        estimatesdexplanterms['alphabeta'] = True
    elif scenarioname in ['Q3ar1', 'Q3studenttsim']:
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1kappa')
    elif scenarioname in ['Q3studenttinference', 'Q3studenttsiminference']:
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1kappa')
        inferenceparams['studenterrors'] = True
        inferenceparams['studenterrors_dof'] = 6            
    else:
        raise NotImplementedError
    return estimateexplanterms, estimatesdexplanterms, inferenceparams

def inference_weights(scenarioname, visible, normalized_weights):
    visible_w = deepcopy(visible)
    if scenarioname in  ['Q1spline', 'Q2spline_base']:
        nsegments = 12
        psp = periodic_spline_predictors(visible['fy'],nsegments=nsegments)
        visible_w['explan']['mu'] = psp['basisfunctions']['diffcoeff']
        visible_w['explan']['lambda'] = psp['basisfunctions']['diffcoeff']
        normalized_weights_w = normalize_weights(explankappa=visible['explan']['kappa'], explanmu=visible_w['explan']['mu'], explanlambda=visible_w['explan']['lambda'], n=len(visible['fy']))
    elif scenarioname == 'Q3logisticspline':
        nsegments = 12
        psp = periodic_spline_predictors(visible['fy'],nsegments=nsegments)
        normalized_weights_w = normalize_weights(explankappa=visible['explan']['kappa'], explanmu=visible_w['explan']['mu'], explanlambda=visible_w['explan']['lambda'], explanalphabeta = psp['basisfunctions']['diffcoeff'],n=len(visible['fy']))
    else:
        normalized_weights_w = deepcopy(normalized_weights)
    return visible_w, normalized_weights_w
    