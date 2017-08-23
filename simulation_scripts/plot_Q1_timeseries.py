'''
Created on Jul 10, 2017

@author: zwieback
'''
import matplotlib.pyplot as plt
from simulation_paths import path
from simulation_analysis import analyse_simulation_path, compute_metrics, results_simulation_path, extract_values
import numpy as np
from plotting import colsgrey, prepare_figure, wrrblues
import datetime
#pathoutrep = os.path.join(path, scenario, str(n), str(rep))

parameters = ['m', 'l', 'sigmap', 'mu', 'lambda', 'kappa', ]


markersize=4
ind_products=[1]
n_points = 151
ind_traces=[101, 301, 501, 233, 72, 546]#[1,101,201,401,501,901,233]#[301,501]#[1,101,201,401]#1,2
#101, 301, 501, 233, 224, 72
show_traces = [1]
xrange=(0,366)
xticks = (1, 92, 183, 275, 366)
xticklabels = ('Jan', 'Apr', 'Jul', 'Oct', 'Jan')
yticks=[(0.1,0.3),(0.0,0.05),(1.0,1.2)]
ylims=[(-0.07,0.45),(0.0,0.075),(0.9,1.38)]
def prepare_offset_terms(results):
    lambdavals = extract_values('lambda',results['trace'], results['internal'])
    lvals = extract_values('l',results['trace'], results['internal'])
    lvals[1][0,:]=1
    muvals = extract_values('mu',results['trace'], results['internal'])
    mvals = extract_values('m',results['trace'], results['internal'])
    mvals[1][0,:]=0
    n_meas=results['normalized_weights']['mu']['weight'].shape[1]
    if len(muvals[0]) == 0:
        Mtrue = mvals[0][:,np.newaxis]*np.ones(n_meas)[np.newaxis,:]
        Ltrue = lvals[0][:,np.newaxis]*np.ones(n_meas)[np.newaxis,:]       
    else:
        Mtrue = mvals[0][:,np.newaxis] + np.sum(muvals[0][:,np.newaxis,np.newaxis]*results['normalized_weights']['mu']['weight'], axis=1)
        Ltrue = lvals[0][...,np.newaxis] + np.sum(lambdavals[0][:,np.newaxis,np.newaxis]*results['normalized_weights']['lambda']['weight'], axis=1)
    if muvals[1].shape[1] == 0:    
        Mtrace = mvals[1][...,np.newaxis]*np.ones(n_meas)[np.newaxis,np.newaxis,:]
        Ltrace = lvals[1][...,np.newaxis]*np.ones(n_meas)[np.newaxis,np.newaxis,:]
    else:    
        Ltrace = lvals[1][...,np.newaxis] + np.sum(lambdavals[1][...,np.newaxis]*results['normalized_weights']['lambda']['weight'][np.newaxis,:,np.newaxis,:], axis=1)
        Mtrace = mvals[1][...,np.newaxis] + np.sum(muvals[1][...,np.newaxis]*results['normalized_weights']['mu']['weight'][np.newaxis,:,np.newaxis,:], axis=1)

    return Mtrue, Mtrace, Ltrue, Ltrace
def ylab(ax,lab1,lab2):
    ax.text(-0.19, 0.5, lab1, transform = ax.transAxes, rotation='vertical',ha='right', va='center')
    ax.text(-0.13, 0.5, lab2, transform = ax.transAxes, rotation='vertical',ha='right', va='center')
def plot_Q1_timeseries(scenarios, n=500, scenariosadd={}):
    fig, axs = prepare_figure(nrows=3, ncols=len(scenarios), figsize_columns = (1.7, 0.85), sharex=True, sharey = 'row', bottom=0.09, left=0.12, right=0.85, top=0.94, hspace=0.2, wspace = 0.2)
    
    import input_output
    import os
    '''
    results = {}
    scenarios0 = scenarios + list(scenariosadd.values())
    for scenario in scenarios0:
        pathscenario = os.path.join(path, scenario, str(n), '0')     
        results[scenario]= results_simulation_path(pathscenario, recursive = False)
    input_output.save_pickle('C:\\Work\\SMAP\\simulations\\test4.p',results)
    '''
    results = input_output.read_pickle('C:\\Work\\SMAP\\simulations\\test4.p')
    
    results2 = results['Q2spline_base']
    extract_values('mu',results2['trace'], results2['internal'])
    
    
    for jscen, scenario in enumerate(scenarios):
        lw = 2.2
        x = results[scenario]['visible']['day']
        xdate = [datetime.datetime(2016,1,1)+datetime.timedelta(int(d)) for d in x]
        y = results[scenario]['visible']['y']
        theta = results[scenario]['internal']['theta']
        axs[0, jscen].plot(x[0:n_points], theta[0:n_points], lw=lw, color = colsgrey[2])
        cols = ['none',wrrblues[0]]
        colse = [colsgrey[0],wrrblues[0]]
        colstrue = [colsgrey[2],wrrblues[0]]
        colstrace = [wrrblues[0]]
        coltraceadd = wrrblues[1]
        lwtrace = 0.1
        mew=0.2
        for jind_product, ind_product in enumerate(ind_products):
            axs[0, jscen].plot(x[0:n_points], y[ind_product,0:n_points].T, marker='.', linestyle='None', markersize=markersize, mew=mew, markerfacecolor=cols[jind_product], markeredgecolor=colse[jind_product])#, markerfacecolor='none', markersize=markersize, mew=0.1,
        #axs[0, jscen].plot(x[0:n_points], y[:,0:n_points].T, lw=lw)
        Mtrue, Mtrace, Ltrue, Ltrace = prepare_offset_terms(results[scenario])
        if scenario in scenariosadd:
            Mtruea, Mtracea, Ltruea, Ltracea = prepare_offset_terms(results[scenariosadd[scenario]]) 
        for jind_product, ind_product in enumerate(ind_products):            
            axs[1, jscen].plot(x[0:n_points],Mtrue[ind_product,0:n_points], c = colstrue[jind_product], lw=lw)
            axs[2, jscen].plot(x[0:n_points],Ltrue[ind_product,0:n_points], c = colstrue[jind_product], lw=lw)
            if ind_product in show_traces:
                axs[1, jscen].plot(x[0:n_points],Mtrace[ind_traces,ind_product,0:n_points].T, lw=lwtrace, c=colstrace[jind_product])
                axs[2, jscen].plot(x[0:n_points],Ltrace[ind_traces,ind_product,0:n_points].T, lw=lwtrace, c=colstrace[jind_product])
                if scenario in scenariosadd:
                    axs[1, jscen].plot(x[0:n_points],Mtracea[ind_traces,ind_product,0:n_points].T, lw=lwtrace, c=coltraceadd)
                    axs[2, jscen].plot(x[0:n_points],Ltracea[ind_traces,ind_product,0:n_points].T, lw=lwtrace, c=coltraceadd)
        axs[2, jscen].set_xlim(xrange)
        axs[2, jscen].set_xticks(xticks)
        axs[2, jscen].set_xticklabels(xticklabels)
        
        for jrow in np.arange(3):
            axs[jrow, jscen].set_ylim(ylims[jrow])
            axs[jrow, jscen].set_yticks(yticks[jrow])

    smline, = axs[0, len(scenarios)-1].plot([], [], lw=lw, color = colsgrey[2])
    jind_product=0
    frameon=False
    borderaxespad=1.2
    handlelength=0.8
    handletextpad=0.4
    fontsize=8
    smprod, = axs[0, len(scenarios)-1].plot([], [], marker='.', linestyle='None', markersize=markersize, mew=mew, markerfacecolor=cols[jind_product], markeredgecolor=colse[jind_product])#, markerfacecolor='none', markersize=markersize, mew=0.1,
    axs[0, len(scenarios)-1].legend(handles=[smline,smprod],labels=['true','observed'],loc='center left',bbox_to_anchor=(1.00, 0.5),frameon=frameon,borderaxespad=borderaxespad,handlelength=handlelength,handletextpad=handletextpad,fontsize=fontsize)        
    ylab(axs[0, 0],'Soil moisture','[$\\mathrm{m}^3$ $\\mathrm{m}^{-3}$]')
    trueline, = axs[1, len(scenarios)-1].plot([],[], c = colstrue[jind_product], lw=lw)
    traceline, = axs[1, len(scenarios)-1].plot([],[], lw=lwtrace, c=colstrace[jind_product])
    traceaddline, = axs[1, len(scenarios)-1].plot([],[], lw=lwtrace, c=coltraceadd)
    axs[1, len(scenarios)-1].legend(handles=[trueline,traceline,traceaddline],labels=['true','estimated','spline fit'],loc='center left',bbox_to_anchor=(1.00, 0.5),frameon=frameon,borderaxespad=borderaxespad,handlelength=handlelength,handletextpad=handletextpad,fontsize=fontsize)
    ylab(axs[1, 0],'$M$ (additive)','[$\\mathrm{m}^3$ $\\mathrm{m}^{-3}$]')
    trueline, = axs[2, len(scenarios)-1].plot([],[], c = colstrue[jind_product], lw=lw)
    traceline, = axs[2, len(scenarios)-1].plot([],[], lw=lwtrace, c=colstrace[jind_product])
    traceaddline, = axs[2, len(scenarios)-1].plot([],[], lw=lwtrace, c=coltraceadd)
    axs[2, len(scenarios)-1].legend(handles=[trueline,traceline,traceaddline],labels=['true','estimated','spline fit'],loc='center left',bbox_to_anchor=(1.00, 0.5),frameon=frameon,borderaxespad=borderaxespad,handlelength=handlelength,handletextpad=handletextpad,fontsize=fontsize)
    ylab(axs[2, 0],'$L$ (multiplicative)','[-]')
    for ax in axs.flatten():
        for loc, spine in ax.spines.items():
            if loc in ['right','top']:
                spine.set_color('none')
    #plt.show()
    plt.savefig('C:\\Work\\SMAP\\simulations\\Q1_ts.pdf')
def plot_Q1_timeseries_onecolumn(scenario, n=500, scenariosadd={}):
    fig, axs = prepare_figure(nrows=3, ncols=1, figsize_columns = (1.3, 0.85), sharex=True, sharey = 'row', bottom=0.09, left=0.15, right=0.78, top=0.94, hspace=0.2, wspace = 0.2)
    
    import input_output
    import os
    '''
    results = {}
    scenarios0 = [scenario] + list(scenariosadd.values())
    for scenario in scenarios0:
        pathscenario = os.path.join(path, scenario, str(n), '0')     
        results[scenario]= results_simulation_path(pathscenario, recursive = False)
    input_output.save_pickle('C:\\Work\\SMAP\\simulations\\test4.p',results)
    '''
    results = input_output.read_pickle('C:\\Work\\SMAP\\simulations\\test4.p')
    
    
    lw = 2.2
    alphatrac = 0.7
    alphatracadd = 0.4
    x = results[scenario]['visible']['day']
    xdate = [datetime.datetime(2016,1,1)+datetime.timedelta(int(d)) for d in x]
    y = results[scenario]['visible']['y']
    theta = results[scenario]['internal']['theta']
    axs[0].plot(x[0:n_points], theta[0:n_points], lw=lw, color = colsgrey[2])
    cols = ['none',wrrblues[0]]
    colse = [colsgrey[0],wrrblues[0]]
    colstrue = [colsgrey[2],wrrblues[0]]
    colstrace = [wrrblues[0]]
    coltraceadd = wrrblues[1]
    lwtrace = 0.1
    lwtraceadd = 0.1
    mew=0.2
    for jind_product, ind_product in enumerate(ind_products):
        axs[0].plot(x[0:n_points], y[ind_product,0:n_points].T, marker='.', linestyle='None', markersize=markersize, mew=mew, markerfacecolor=cols[jind_product], markeredgecolor=colse[jind_product])#, markerfacecolor='none', markersize=markersize, mew=0.1,
    #axs[0, jscen].plot(x[0:n_points], y[:,0:n_points].T, lw=lw)
    Mtrue, Mtrace, Ltrue, Ltrace = prepare_offset_terms(results[scenario])
    if scenario in scenariosadd:
        Mtruea, Mtracea, Ltruea, Ltracea = prepare_offset_terms(results[scenariosadd[scenario]]) 
    for jind_product, ind_product in enumerate(ind_products):            
        axs[1].plot(x[0:n_points],Mtrue[ind_product,0:n_points], c = colstrue[jind_product], lw=lw)
        axs[2].plot(x[0:n_points],Ltrue[ind_product,0:n_points], c = colstrue[jind_product], lw=lw)
        if ind_product in show_traces:
            axs[1].plot(x[0:n_points],Mtrace[ind_traces,ind_product,0:n_points].T, lw=lwtrace, c=colstrace[jind_product], alpha = alphatrac)
            axs[2].plot(x[0:n_points],Ltrace[ind_traces,ind_product,0:n_points].T, lw=lwtrace, c=colstrace[jind_product], alpha = alphatrac)
            #print(ind_product)
            #print(np.nonzero(np.max(Mtrace[:,ind_product,:],axis=1)>0.073))
            if scenario in scenariosadd:
                axs[1].plot(x[0:n_points],Mtracea[ind_traces,ind_product,0:n_points].T, lw=lwtraceadd, c=coltraceadd, alpha = alphatracadd)
                axs[2].plot(x[0:n_points],Ltracea[ind_traces,ind_product,0:n_points].T, lw=lwtraceadd, c=coltraceadd, alpha = alphatracadd)
    axs[2].set_xlim(xrange)
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticklabels)
    
    for jrow in np.arange(3):
        axs[jrow].set_ylim(ylims[jrow])
        axs[jrow].set_yticks(yticks[jrow])

    smline, = axs[0].plot([], [], lw=lw, color = colsgrey[2])
    jind_product=0
    frameon=False
    borderaxespad=1.2
    handlelength=0.8
    handletextpad=0.4
    fontsize=8
    smprod, = axs[0].plot([], [], marker='.', linestyle='None', markersize=markersize, mew=mew, markerfacecolor=cols[jind_product], markeredgecolor=colse[jind_product])#, markerfacecolor='none', markersize=markersize, mew=0.1,
    axs[0].legend(handles=[smline,smprod],labels=['true','observed'],loc='center left',bbox_to_anchor=(1.00, 0.5),frameon=frameon,borderaxespad=borderaxespad,handlelength=handlelength,handletextpad=handletextpad,fontsize=fontsize)        
    ylab(axs[0],'Soil moisture','[$\\mathrm{m}^3$ $\\mathrm{m}^{-3}$]')
    trueline, = axs[1].plot([],[], c = colstrue[jind_product], lw=lw)
    traceline, = axs[1].plot([],[], lw=lwtrace, c=colstrace[jind_product], alpha=alphatrac)
    traceaddline, = axs[1].plot([],[], lw=lwtraceadd, c=coltraceadd, alpha=alphatracadd)
    axs[1].legend(handles=[trueline,traceline,traceaddline],labels=['true','parametric fit','spline fit'],loc='center left',bbox_to_anchor=(1.00, 0.5),frameon=frameon,borderaxespad=borderaxespad,handlelength=handlelength,handletextpad=handletextpad,fontsize=fontsize)
    ylab(axs[1],'$M$ (additive)','[$\\mathrm{m}^3$ $\\mathrm{m}^{-3}$]')
    trueline, = axs[2].plot([],[], c = colstrue[jind_product], lw=lw)
    traceline, = axs[2].plot([],[], lw=lwtrace, c=colstrace[jind_product], alpha=alphatrac)
    traceaddline, = axs[2].plot([],[], lw=lwtrace, c=coltraceadd, alpha=alphatracadd)
    axs[2].legend(handles=[trueline,traceline,traceaddline],labels=['true','parametric fit','spline fit'],loc='center left',bbox_to_anchor=(1.00, 0.5),frameon=frameon,borderaxespad=borderaxespad,handlelength=handlelength,handletextpad=handletextpad,fontsize=fontsize)
    ylab(axs[2],'$L$ (multiplicative)','[-]')
    for ax in axs.flatten():
        for loc, spine in ax.spines.items():
            if loc in ['right','top']:
                spine.set_color('none')
    #plt.show()
    plt.savefig('C:\\Work\\SMAP\\simulations\\Q1_ts.pdf')    
if __name__ == '__main__':
    #plot_Q1_timeseries(['Q1base','Q1lambdamu'],scenariosadd={'Q1lambdamu':'Q1spline', 'Q1base':'Q2spline_base'})    #,'Q1kappa'
    plot_Q1_timeseries_onecolumn('Q1lambdamu',scenariosadd={'Q1lambdamu':'Q1spline'})