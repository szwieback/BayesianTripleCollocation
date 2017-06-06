'''
Created on Jun 1, 2017

@author: zwieback
'''
import numpy as np
from scipy.stats import gmean

# normalize explanatory variables such that their product is 1 (i.e. divide by their geometric mean)
def normalize_weight_multiplicative(explanatory):
    # weight is assumed positive
    assert np.count_nonzero(explanatory<0)==0
    multfactor=gmean(explanatory,axis=1)    
    nweight=explanatory/multfactor[:,np.newaxis]
    return nweight, {'multiplicative':multfactor}

# normalize explanatory variables so that they have mean zero and standard deviation 1
def normalize_weight_additive(explanatory):
    addfactor = np.mean(explanatory,axis=1)
    weight = explanatory-addfactor[:,np.newaxis]
    multfactor=np.std(weight,axis=1)
    nweight = weight / multfactor[:,np.newaxis]
    return nweight, {'multiplicative':multfactor,'additive':addfactor}

def prepare_weights(explan=None,n=None,multiplicative=False,additive=True):
    assert multiplicative != additive #either mult. or add.
    if explan is None:
        assert n is not None
        explan = np.zeros((0,n))
    nfac=explan.shape[0]
    if multiplicative:
        weight,normfactor = normalize_weight_multiplicative(explan)
    elif additive:
        weight,normfactor = normalize_weight_additive(explan)
    else:
        raise
    return {'nfac':nfac,'normfactor':normfactor,'weight':weight}    

def normalize_weights(explankappa=None,explanmu=None,explanlambda=None,explanalphabeta=None,n=None):
    explanatoryweights={}    
    # prepare kappa weights
    explanatoryweights['kappa']=prepare_weights(explan=explankappa,n=n,multiplicative=True,additive=False)
    
    # prepare mu weights
    explanatoryweights['mu']=prepare_weights(explan=explanmu,n=n,multiplicative=False,additive=True)
    
    # prepare lambda weights
    explanatoryweights['lambda']=prepare_weights(explan=explanlambda,n=n,multiplicative=False,additive=True)
    
    # prepare alpha/beta weights
    explanatoryweights['alphabeta']=prepare_weights(explan=explanalphabeta,n=n,multiplicative=False,additive=True)

    return explanatoryweights