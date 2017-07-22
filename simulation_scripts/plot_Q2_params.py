'''
Created on Jul 5, 2017

@author: zwieback
'''
import matplotlib.pyplot as plt
from simulation_paths import path
from simulation_analysis import analyse_simulation_path, compute_metrics
from plotting import prepare_figure, draw_notches, draw_dot, globfigparams, colsgrey
import numpy as np
#pathoutrep = os.path.join(path, scenario, str(n), str(rep))
horlims = {'sigmap':(0,0.015),'mu':(0,0.09),'lambda':(0,0.1),'kappa':(0,0.4)}
xticks = {'sigmap': (0, 0.01), 'mu':(0,0.05), 'lambda': (0,0.05), 'kappa': (0,0.2)}
skipbars = [('Q2kappa_base','kappa'),('Q2kappa_base','lambda'),('Q2kappa_base','mu'),('Q2kappa_lambdamu','kappa')]
clipfirstvalues = ['mu','lambda','kappa']
xlabels = {'sigmap': '$\\sigma$ [$\\mathrm{m}^3$ $\\mathrm{m}^{-3}$]', 'mu': '$\\mu$ [-]', 'lambda': '$\\lambda$ [-]', 'kappa':'$\\kappa$ [-]'}
coltitles = {'sigmap': 'Error $\\sigma$', 'lambda': 'Multiplicative $\\lambda$', 'kappa': 'Noise coeff. $\\kappa$', 'mu': 'Additive $\\mu$'}
rowlabels = {'Q2kappa_base': 'No $\\mu$, $\\lambda$, $\\kappa$', 'Q1kappa': 'Full model', 'Q2kappa_lambdamu': 'No $\\kappa$'}



def plot_Q2_params(parameters, scenarios = ['Q2kappa_base', 'Q2lambdamu_base', 'Q2kappa_lambdamu'], ns = [100,250,500], horlims = {}):
    ncols = len(parameters)    
    fig, axs = prepare_figure(nrows=len(scenarios), ncols=ncols, figsize_columns=(1.7, 0.8), sharex='col', sharey=True, bottom=0.145, left=0.05, right=0.81, top=0.925, hspace=0.45)
    
    import input_output
    import os
    '''
    metrics = {}
    for scenario in scenarios:
        metrics[scenario]={}
        for parameter in parameters:
            metrics[scenario][parameter]={}
            for n in ns:
                summary = analyse_simulation_path(os.path.join(path, scenario, str(n)), params = parameters, recursive = True)
                metrics[scenario][parameter][n] = compute_metrics(summary, parameter, clipfirstvalue = parameter in clipfirstvalues)
    input_output.save_pickle('C:\\Work\\SMAP\\simulations\\test2.p',metrics)
    '''
    metrics = input_output.read_pickle('C:\\Work\\SMAP\\simulations\\test2.p')
    metricnamebar = 'absolute_uncertainty_empirical'
    metricnameline = 'absolute_bias_magnitude'
    metricdot = 'absolute_uncertainty_posterior'
    x = np.arange(len(ns))
    height = 0.8
    for jscen,scenario in enumerate(scenarios):
        axs[jscen,0].set_ylabel(rowlabels[scenario], size=9, labelpad=10)
        for jparam,parameter in enumerate(parameters):
            if parameter in horlims:
                axs[jscen,jparam].set_xlim(horlims[parameter])
                axs[jscen,jparam].set_ylim((-0.5,len(ns)-0.5))
            ybar = np.array([metrics[scenario][parameter][n][metricnamebar] for n in ns])
            yline = np.array([metrics[scenario][parameter][n][metricnameline] for n in ns])
            ydot = np.array([metrics[scenario][parameter][n][metricdot] for n in ns])
            #y1 = np.array([metrics[scenario][n][rowmetrics[1]] for n in ns])
            if (scenario,parameter) not in skipbars:
                axs[jscen,jparam].barh(x, ybar, height, color=colsgrey)
            else:
                axs[jscen,jparam].text(0.06,0,'set to 0',va='bottom',ha='left',transform=axs[jscen,jparam].transAxes)
            for jn,n in enumerate(ns):
                draw_notches(axs[jscen,jparam],yline[jn],x[jn],height=height)
                draw_dot(axs[jscen,jparam],ydot[jn],x[jn])
            axs[jscen,jparam].set_yticks([])
            if parameter in xticks:
                axs[jscen,jparam].set_xticks(xticks[parameter])
    for jparam,parameter in enumerate(parameters):       
        axs[len(scenarios)-1,jparam].set_xlabel(xlabels[parameter])
        axs[0,jparam].set_title(coltitles[parameter],size=9)
        #axs[jpanel].set_title(titles[jpanel])
    '''
    for jrow in range(2):
        axs[jrow,0].set_ylabel(rowlabels[jrow])
    '''
    xlabel = 0.42
    ax=axs[0,len(parameters)-1]
    for jn, n in enumerate(ns):
        ax.text(xlabel,jn,str(n)+' samples',transform=ax.transData, va='center', ha='left')
    #ax.text(xlabel, jn + 0.85, 'Sample size', transform=ax.transData, va='center', ha='left')
    #ax.text(xlabel+0.1, (len(ns)-1)/2, 'Sample size', transform=ax.transData, va='center', ha='left', rotation=-90)
    
    # separate legend
    heightax=axs[0,0].get_position().y1-axs[0,0].get_position().y0
    widthax=axs[0,0].get_position().x1-axs[0,0].get_position().x0
    axleg = fig.add_axes((0.875,0.4,widthax,heightax))#-0.01
    axleg.set_ylim(axs[0,0].get_ylim())
    for loc, spine in axleg.spines.items():
        spine.set_color('none')
    axleg.set_xticks([])
    axleg.set_yticks([])
    axleg.set_xlim((0,1))
    barl=0.25
    biasl=0.1
    dotl=0.4
    axleg.barh(1, barl, height, color='#dddddd')
    draw_notches(axleg,biasl,1,height=height)
    draw_dot(axleg, dotl, 1)
    axleg.annotate('Bias', xy=(biasl,1-0.5*height), xytext=(biasl-0.02,0.3),ha='center',va='top')
    axleg.plot((biasl, biasl-0.01),(1-0.5*height-0.1, 0.35),lw=plt.rcParams['axes.linewidth'],c=globfigparams['fontcolour'])
    axleg.annotate('RMSE', xy=(barl,1+0.5*height), xytext=(barl+0.00,1.7),ha='center',va='bottom')
    axleg.plot((barl, barl),(1+0.5*height+0.1, 1.67),lw=plt.rcParams['axes.linewidth'],c=globfigparams['fontcolour'])
    axleg.annotate('Posterior', xy=(dotl, 1-0.5*height), xytext=(dotl+0.1,0.3), ha='center', va='top')
    axleg.plot((dotl+0.02, dotl+0.04),(1-0.3, 0.35),lw=plt.rcParams['axes.linewidth'],c=globfigparams['fontcolour'])
    
    '''
    # Annotate one plot
    axs[len(scenarios)-1,len(parameters)-1].text(yline[0]+0.04,len(ns)-0.5,'Bias',va='bottom',ha='center')
    axs[len(scenarios)-1,len(parameters)-1].plot((yline[0]+0.015,yline[0]+0.035),(len(ns)-0.85,len(ns)-0.5),lw=plt.rcParams['axes.linewidth'],c=globfigparams['fontcolour'])
    axs[len(scenarios)-1,len(parameters)-1].text(ydot[0]+0.04,len(ns)-0.5,'Posterior',va='bottom',ha='center')
    axs[len(scenarios)-1,len(parameters)-1].plot((ydot[0]+0.015,ydot[0]+0.035),(len(ns)-0.85,len(ns)-0.5),lw=plt.rcParams['axes.linewidth'],c=globfigparams['fontcolour'])
    '''
    
    for ax in axs.flatten():
        for loc, spine in ax.spines.items():
            if loc in ['right','top','left']:
                spine.set_color('none')
    #plt.show()
    plt.savefig('C:\\Work\\SMAP\\simulations\\Q12.pdf')
if __name__ == '__main__':
    parameters = ['sigmap', 'mu', 'lambda', 'kappa']
    plot_Q2_params(parameters, scenarios = ['Q1kappa','Q2kappa_lambdamu','Q2kappa_base'], horlims=horlims)