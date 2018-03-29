'''
Created on Feb 27, 2018

@author: zwieback
'''
# sigma: exponential: mean 0.1^2
#m and mu: T(0,0.3^2;4)
#l : T(1,0.3^2;4)
import numpy as np
from scipy.stats import t, norm
import matplotlib.pyplot as plt
from plotting import colsgrey,prepare_figure,globfigparams


labels = ['$\\sigma^2$ $[(\\mathrm{m}^3 \\, \\mathrm{m}^{-3})^2]$', '$m$ and $\\mu$ [$\\mathrm{m}^3 \\, \\mathrm{m}^{-3}$]', '$l$ [-]', '$\\lambda$ [-]']

fig, axs = prepare_figure(nrows=1, ncols=4, figsize_columns=(1.9, 0.5), sharex=False, sharey=False, bottom=0.22, left=0.02, right=0.985, top=0.92, hspace=0.39)
lw=1.0
col = 'k'
xsigma2 = np.linspace(0,0.5)
axs[0].plot(xsigma2,np.exp(-xsigma2/(0.1)), lw=lw, c=col)

xm = np.linspace(-0.7,0.7)
df=4
scale = 0.3/t.std(df=df)
dist = t(df=df,scale=scale)
axs[1].plot(xm, dist.pdf(xm), lw=lw, c=col)
axs[3].plot(xm, dist.pdf(xm), lw=lw, c=col)

xl = np.linspace(0,2.0)
df=4
scale = 0.3/t.std(df=df)
dist = t(df=df,scale=scale,loc=1.0)
axs[2].plot(xl, dist.pdf(xl), lw=lw, c=col)

for j,ax in enumerate(axs.flatten()):
    for loc, spine in ax.spines.items():
        if loc in ['right','top','left']:
            spine.set_color('none')    
            ax.set_yticks([])
            ylim=(0,ax.get_ylim()[1])
            ax.set_ylim(ylim)           
            ax.text(0.5,-0.2,labels[j],ha='center', va='top', transform=ax.transAxes)#, color=globfigparams['fontcolour'])
ax.text(0.5,0.98,'Prior distributions',ha='center',va='top', transform=fig.transFigure, size=8, color='k')            
plt.savefig('C:\\Work\\SMAP\\simulations\\priors.pdf')