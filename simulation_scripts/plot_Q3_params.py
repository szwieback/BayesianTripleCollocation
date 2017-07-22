'''
Created on Jul 7, 2017

@author: zwieback
'''
import matplotlib.pyplot as plt
from simulation_paths import path
from simulation_analysis import analyse_simulation_path, compute_metrics
import numpy as np
from plotting import draw_notches, prepare_figure, colsgrey


clipfirstvalues = ['mu','lambda','kappa']
scenariolabels = {'Q1kappa':'baseline','Q3dof':'prior tails', 'Q3priorfactor':'prior scale', 'Q3beta':'moisture marginal', 'Q3logisticspline':'moisture seasonal', 'Q3ar1':'autocorrelation', 'Q3studenttsim':'t error simulation', 'Q3studenttinference':'t error inference', 'Q3studenttsiminference':'t error'}
xlabels = {'sigmap': '$\\sigma$ [$\\mathrm{m}^3$ $\\mathrm{m}^{-3}$]', 'mu': '$\\mu$ [-]', 'lambda': '$\\lambda$ [-]', 'kappa':'$\\kappa$ [-]'}
coltitles = {'sigmap': 'Error $\\sigma$', 'lambda': 'Multiplicative $\\lambda$', 'kappa': 'Noise coeff. $\\kappa$', 'mu': 'Additive $\\mu$'}
def plot_Q3_params(parameters, scenarios = ['Q3dof', 'Q3priorfactor', 'Q3beta', 'Q3logisticspline', 'Q3ar1', 'Q3studenttsim', 'Q3studenttinference', 'Q3studenttsiminference'], n=250, horlims = {}):
    ncols = len(parameters)
    sharex='col'
    sharey=True
    figsize_columns = (1.7, 0.65)
    fig, axs = prepare_figure(nrows = 1, ncols=ncols, figsize_columns=figsize_columns, sharex=sharex, sharey=sharey, bottom=0.18,left=0.18, right=0.97,top=0.91,hspace=0.45)
    import input_output
    import os
    '''
    metrics = {}
    for scenario in scenarios:
        metrics[scenario]={}
        summary = analyse_simulation_path(os.path.join(path, scenario, str(n)), params = parameters, recursive = True)
        for parameter in parameters:
            metrics[scenario][parameter]={}            
            metrics[scenario][parameter][n] = compute_metrics(summary, parameter, clipfirstvalue = parameter in clipfirstvalues)
    input_output.save_pickle('C:\\Work\\SMAP\\simulations\\test3.p',metrics)
    '''
    metrics = input_output.read_pickle('C:\\Work\\SMAP\\simulations\\test3.p')
    metricnamebar = 'absolute_uncertainty_empirical'#'absolute_bias_magnitude'##, 'relative_bias_magnitude']
    metricnameline = 'absolute_bias_magnitude'    
    
    sortparameter = parameters[0]
    ybaruns = np.array([metrics[scenario][sortparameter][n][metricnamebar] for scenario in scenarios])[1:]
    x = np.concatenate([[0],np.argsort(ybaruns)+1])
    scenariossort = np.flip(np.array(scenarios)[x],axis=0)
    gridlines = np.arange(0,len(scenarios)-1,4)

    x = np.arange(len(scenarios))
    colsbars = [colsgrey[2]]*(len(scenarios)-1)+[colsgrey[1]]

    height = 0.8       
    for jparam,parameter in enumerate(parameters):
        if parameter in horlims:
            axs[jparam].set_xlim(horlims[parameter])
            axs[jparam].set_ylim((-0.5,len(scenarios)-0.5))
        ybar = np.array([metrics[scenario][parameter][n][metricnamebar] for scenario in scenariossort])
        yline = np.array([metrics[scenario][parameter][n][metricnameline] for scenario in scenariossort])
        axs[jparam].barh(x, ybar, height, color=colsbars)    
        axs[jparam].set_yticks(x)
        axs[jparam].set_yticklabels([scenariolabels[scenario] for scenario in scenariossort], rotation='horizontal')
        for tic in axs[jparam].yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        for jscenario,scenario in enumerate(scenariossort):
            draw_notches(axs[jparam],yline[jscenario],x[jscenario],height=height,axheighttriangle=0.03)
        
        #for gridline in gridlines:
        #    axs[jparam].axhline(x[gridline]+0.5, linestyle = '-', color='#dddddd', linewidth=0.25)
    for jparam,parameter in enumerate(parameters):       
        axs[jparam].set_xlabel(xlabels[parameter])
        axs[jparam].set_title(coltitles[parameter],size=9)       

    for ax in axs.flatten():
        for loc, spine in ax.spines.items():
            if loc in ['right','top','left']:
                spine.set_color('none')    
    
    #plt.show()
    plt.savefig('C:\\Work\\SMAP\\simulations\\Q3.pdf')
    
if __name__ == '__main__':
    parameters = ['sigmap', 'mu', 'lambda', 'kappa']
    plot_Q3_params(parameters, scenarios = ['Q1kappa','Q3dof', 'Q3priorfactor', 'Q3beta', 'Q3logisticspline', 'Q3ar1', 'Q3studenttsim', 'Q3studenttinference', 'Q3studenttsiminference'])    