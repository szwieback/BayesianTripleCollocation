'''
Created on Jun 27, 2017

@author: zwieback
'''
import numpy as np
from functools import reduce
numpy_rng = np.random.RandomState(123456)
var = 0.6
n = 1500
noisewhite = np.sqrt(var)*numpy_rng.standard_normal(size=(n))
noiseinnov = np.sqrt(var)*numpy_rng.standard_normal(size=(n-1))
#mixing with 1/2 (previous final noise term) and sqrt(3)/2 (current noise term before mixing)
noise = noisewhite.copy()
weightprev = 0.5
weightinnov = np.sqrt(3)/2
for j in np.arange(1, n):
    noise[j] = weightprev*noise[j-1] + weightinnov*noiseinnov[j-1]
print(np.std(noise)**2)
print(np.mean(noise[1:]*noise[:-1])/np.var(noise))