'''
Created on Jul 5, 2017

@author: zwieback
'''

import matplotlib.pyplot as plt
from simulation_paths import path
from simulation_analysis import analyse_simulation_path, compute_metrics
import numpy as np
from plotting import colsgrey,prepare_figure
#pathoutrep = os.path.join(path, scenario, str(n), str(rep))


def plot_Q1_sigmap(scenarios, ns = [100,250,500]):
    ncols = len(scenarios)
    fig, axs = prepare_figure(nrows=2, ncols=ncols, figsize_columns=(1.0, 0.55), sharex=True, sharey=False, bottom=0.23, left=0.08, right=0.885, top=0.89, hspace=0.39)
    import input_output
    import os
    '''
    metrics = {}
    for scenario in scenarios:
        metrics[scenario]={}
        for n in ns:
            summary = analyse_simulation_path(os.path.join(path, scenario, str(n)), recursive = True)
            metrics[scenario][n] = compute_metrics(summary, 'sigmap', clipfirstvalue=False)
    input_output.save_pickle('C:\\Work\\SMAP\\simulations\\test.p',metrics)
    '''
    metrics = input_output.read_pickle('C:\\Work\\SMAP\\simulations\\test.p')
    rowmetrics = ['relative_uncertainty_empirical', 'relative_uncertainty_posterior', 'relative_bias_magnitude']
    rowlabels = ['Empirical', 'Posterior']
    x = np.arange(len(ns))
    
    for jscen,scenario in enumerate(scenarios):
        y0 = np.array([metrics[scenario][n][rowmetrics[0]] for n in ns])
        y1 = np.array([metrics[scenario][n][rowmetrics[1]] for n in ns])
        axs[0,jscen].barh(x, y0*100, 0.8, color=colsgrey)
        axs[1,jscen].barh(x, y1*100, 0.8, color=colsgrey)
        axs[0,jscen].set_title(scenario)
        axs[0,jscen].set_yticks([])
        axs[1,jscen].set_yticks([])
        axs[1,jscen].set_xlabel('Relative error (\%)')
        #axs[jpanel].set_title(titles[jpanel])
    for jrow in range(2):
        axs[jrow,0].set_ylabel(rowlabels[jrow])
    xlabel = 15
    for jn, n in enumerate(ns):
        axs[1,jscen].text(xlabel,jn,str(n),transform=axs[1,jscen].transData, va='center', ha='center')
    axs[1,jscen].text(xlabel, jn + 0.7, 'n', transform=axs[1,jscen].transData, va='center', ha='center')
    for ax in axs.flatten():
        for loc, spine in ax.spines.items():
            if loc in ['right','top','left']:
                spine.set_color('none')    
    #plt.show()
    plt.savefig('C:\\Work\\SMAP\\simulations\\Q1_sigmap.pdf')
if __name__ == '__main__':
    plot_Q1_sigmap(['Q1base','Q1lambdamu','Q1kappa'])