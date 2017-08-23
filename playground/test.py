'''
Created on Jun 7, 2017

@author: zwieback
'''
from input_output import read_results
pathout='C:\\Work\\SMAP\\simulations\\Q2spline_base\\250\\0\\'
import pymc3 as pm
import numpy as np

trace = read_results(pathout)['trace']
#pm.summary(trace,varnames=['sigmap','muest','lambdaest','mest','lest'])
pm.summary(trace,varnames=['muest'])
'''
mu=np.average(trace['muest'],axis=0)
visible = read_results(pathout)['visible']
normalized_weights = read_results(pathout)['normalized_weights']
#print(normalized_weights['mu']['weight'].shape)
mupred=np.tensordot(normalized_weights['mu']['weight'],mu[:,0],axes=([0],[0]))
pathout='C:\\Work\\SMAP\\simulations\\Q1lambdamu\\500\\2\\'
normalized_weights = read_results(pathout)['normalized_weights']
mu1=0.05
import pylab as plt
fig,ax = plt.subplots()
#ax.plot(visible['fy'],visible['explan']['kappa'][0,:])
ax.plot(visible['fy'],mupred,'g')
ax.plot(visible['fy'],normalized_weights['mu']['weight'][0,:]*mu1,'k')
plt.show()
'''
