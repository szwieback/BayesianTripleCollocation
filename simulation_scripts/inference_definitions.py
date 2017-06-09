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
    elif scenarioname == 'Q1spline':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1lambdamu')
        estimatesdexplanterms['mu'] = estimatesdexplanterms['lambda'] = True
    elif scenarioname == 'Q2kappa_lambdamu':
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1lambdamu')
    elif scenarioname in ['Q2kappa_base', 'Q2lambdamu_base']:
        estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition('Q1base')
    else:
        raise NotImplementedError
    return estimateexplanterms, estimatesdexplanterms, inferenceparams

def inference_weights(scenarioname, visible, normalized_weights):
    visible_w = deepcopy(visible)
    if scenarioname == 'Q1spline':
        nsegments = 12
        psp = periodic_spline_predictors(visible['fy'],nsegments=nsegments)
        visible_w['explan']['mu'] = psp['basisfunctions']['diffcoeff']
        visible_w['explan']['lambda'] = psp['basisfunctions']['diffcoeff']
        normalized_weights_w = normalize_weights(explankappa=visible['explan']['kappa'], explanmu=visible_w['explan']['mu'], explanlambda=visible_w['explan']['lambda'], n=len(visible['fy']))
    else:
        normalized_weights_w = deepcopy(normalized_weights)
    return visible_w, normalized_weights_w
    