'''
Created on Jun 2, 2017

@author: zwieback
'''
import numpy as np
from scipy.interpolate import BSpline

def fractional_year(date):
    # from 0 (1 Jan) to 1 (31 Dec); leap year/edge effect handling not perfect 
    try:
        return np.array([np.min([1.0,date1.timetuple().tm_yday/365.25]) for date1 in date])
    except:
        return np.min([1.0,date[0].timetuple().tm_yday/365.25])

def periodic_bspline_basis(fractionalyear,nsegments=12,k=3):
    # periodic bspline representation, parameterized in terms of the "coefficients", i.e. essentially (up to a constant O(1)) the values at the knots
    # returns the basis functions evaluated at fractionalyear
    #fractionalyear: array of fractional_year (between 0 and 1)
    #k: order of bspline; hard-coded to three
    assert k==3
    # evenly spaced points from zero to one
    segmentpoints=np.linspace(0,1,nsegments+1,endpoint=True)
    # segment length
    dsegment=segmentpoints[1]-segmentpoints[0]
    # set up knots for bspline so that periodic bc can be enforced
    knotsend=np.linspace(dsegment,(k)*dsegment,k) # k evenly spaced knots either side
    knots=np.hstack((-knotsend[::-1],segmentpoints,knotsend+1))
    nknots=len(knots)

    # raw basis functions without boundary conditions
    basis_functions=BSpline(knots,np.eye(nknots),k=k)
    Bxraw=basis_functions(fractionalyear)

    # enforce boundary conditions
    Bx=Bxraw[:,1:nsegments+1]
    Bx[:,0]=Bx[:,0]+Bxraw[:,nsegments+1]
    Bx[:,1]=Bx[:,1]+Bxraw[:,nsegments+2]
    Bx[:,nsegments-1]=Bx[:,nsegments-1]+Bxraw[:,0]
    
    return Bx

def mapping_bspline_parameterizations(nsegments=12,k=3):
    # set up non-bijective mapping between coeff and diffcoeff 
    # the constant term is in the nullspace of this mapping, i.e. this is an overparamterized mapping
    # first-order difference because second-order difference mapping has rank deficit of two for even nsegments (consequence of it being anti-symmetric)
    M=(np.diag(np.ones(nsegments),k=0)-np.diag(np.ones(nsegments-1),k=-1))
    M[0,nsegments-1]=-1
    # inverse mapping (nullspace of M is mapped to zero)
    Minv=np.linalg.pinv(M)
    return M,Minv

def periodic_bspline_basis_diffcoeff(fractionalyear,nsegments=12,k=3):
    # basis functions when periodic bspline is parameterized in terms of diffcoeffs [first-order difference between values at adjacent knots, i.e. measure of local slope]
    # good for regularized (smooth) curves
    # constant term has to be specified separately in regression
    M,Minv=mapping_bspline_parameterizations(nsegments=nsegments,k=k)
    Bx=periodic_bspline_basis(fractionalyear,nsegments=nsegments)
    Bxdiffcoeff=np.dot(Bx,Minv)
    return Bxdiffcoeff

def periodic_spline_predictors(fractionalyear,nsegments=12,k=3):
    # top-level function; returns dictionary containing basis functions for different parameterizations and function pointers for conversion/evaluation
    M,Minv=mapping_bspline_parameterizations(nsegments=nsegments,k=k)
    Bx=periodic_bspline_basis(fractionalyear,nsegments=nsegments,k=k)
    Bxdiff=periodic_bspline_basis_diffcoeff(fractionalyear,nsegments=nsegments,k=k)
    coefftodiffcoeff=lambda coeff:np.dot(M,coeff)
    diffcoefftocoeff=lambda diffcoeff: np.dot(Minv,diffcoeff)
    evalbspline=lambda coeff:np.tensordot(Bx,coeff,axes=([1],[0]))
    evalbsplinediffcoeff=lambda diffcoeff:np.tensordot(Bxdiff,diffcoeff,axes=([1],[0]))
    outp={'basisfunctions':{'coeff':Bx,'diffcoeff':Bxdiff},
          'conversion':{'coefftodiffcoeff':coefftodiffcoeff,'diffcoefftocoeff':diffcoefftocoeff},
          'evaluation':{'coeff':evalbspline,'diffcoeff':evalbsplinediffcoeff,'M':M,'Minv':Minv}}
    return outp