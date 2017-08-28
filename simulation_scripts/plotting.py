'''
Created on Jul 12, 2017

@author: zwieback
'''

import matplotlib.pyplot as plt

wrrblues = ['#004174','#5898c9', '#b4cfe4']            
colsgrey = ['#333333','#777777','#bbbbbb']
globfigparams={'fontsize':8,'family':'serif','usetex':True,'preamble':'\\usepackage{times}','column_inch':229.8775/72.27,'markersize':24,'markercolour':'#AA00AA','fontcolour':'#666666','tickdirection':'out'}

def prepare_figure(nrows=1, ncols=1, figsize_columns = (1.7, 0.8), sharex='col', sharey = 'row', squeeze=True, bottom=0.1, left=0.15, right=0.95, top=0.95, hspace=0.5, wspace=0.1):    
    plt.rc('font',**{'size':globfigparams['fontsize'],'family':globfigparams['family']})
    plt.rcParams['text.usetex']=globfigparams['usetex']
    plt.rcParams['text.latex.preamble']=globfigparams['preamble']
    plt.rcParams['legend.fontsize']=globfigparams['fontsize']
    plt.rcParams['font.size']=globfigparams['fontsize']
    plt.rcParams['axes.linewidth']=0.5
    plt.rcParams['axes.labelcolor']=globfigparams['fontcolour']
    plt.rcParams['axes.edgecolor']=globfigparams['fontcolour']
    plt.rcParams['xtick.color']=globfigparams['fontcolour']
    plt.rcParams['xtick.direction']=globfigparams['tickdirection']
    plt.rcParams['ytick.direction']=globfigparams['tickdirection']
    plt.rcParams['ytick.color']=globfigparams['fontcolour']
    plt.rcParams['text.color']=globfigparams['fontcolour']  
    width=globfigparams['column_inch']
    
    figprops = dict(facecolor='white',figsize=(figsize_columns[0]*width, figsize_columns[1]*width))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols ,sharex=sharex,sharey=sharey,squeeze=squeeze)
    plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top, hspace=hspace, wspace=wspace)
    fig.set_facecolor(figprops['facecolor'])
    fig.set_size_inches(figprops['figsize'],forward=True)
    return fig, axs

def draw_notches(ax,y,x,height=0.8,axwidthtriangle=0.07,axheighttriangle=0.1,axheighttrianglefudge=0.000, fc='#ffffff', lw=0.4, ec='none', lc=wrrblues[0]):
    height = height    
    pointtopax = (ax.transLimits.transform((y,x+0.5*height)))
    pointbottomax = (ax.transLimits.transform((y,x-0.5*height)))
    ax.plot((pointtopax[0],pointbottomax[0]),(pointtopax[1]-0.1*height,pointbottomax[1]+0.1*height), color=lc, lw=lw, transform=ax.transAxes)
    triangletop = plt.Polygon(([[pointtopax[0],pointtopax[1]-axheighttriangle],[pointtopax[0]-0.5*axwidthtriangle,pointtopax[1]+axheighttrianglefudge],[pointtopax[0]+0.5*axwidthtriangle,pointtopax[1]+axheighttrianglefudge]]), transform=ax.transAxes, fc=fc, ec=lc, lw=lw, zorder=6)
    trianglebottom = plt.Polygon(([[pointbottomax[0],pointbottomax[1]+axheighttriangle],[pointbottomax[0]-0.5*axwidthtriangle,pointbottomax[1]-axheighttrianglefudge],[pointbottomax[0]+0.5*axwidthtriangle,pointbottomax[1]-axheighttrianglefudge]]), transform=ax.transAxes, fc=fc, ec=lc, lw=lw, zorder=6)
    ax.add_patch(triangletop)
    ax.add_patch(trianglebottom)
    #ax.plot((pointtopax[0]-0.5*axwidthtriangle,pointtopax[0]),(pointtopax[1],pointtopax[1]-axheighttriangle), color=lc, lw=lw, transform=ax.transAxes, zorder=7)
    #ax.plot((pointtopax[0]+0.5*axwidthtriangle,pointtopax[0]),(pointtopax[1],pointtopax[1]-axheighttriangle), color=lc, lw=lw, transform=ax.transAxes, zorder=7)
    #ax.plot((pointbottomax[0]-0.5*axwidthtriangle,pointbottomax[0]),(pointbottomax[1],pointbottomax[1]+axheighttriangle), color=lc, lw=lw, transform=ax.transAxes, zorder=7)
    #ax.plot((pointbottomax[0]+0.5*axwidthtriangle,pointbottomax[0]),(pointbottomax[1],pointbottomax[1]+axheighttriangle), color=lc, lw=lw, transform=ax.transAxes, zorder=7)

def draw_dot(ax,y,x,fc='#ffffff', ec=wrrblues[0], lw=0.5, r=6):
    #circle = plt.Circle(ax.transLimits.transform((y,x)), radius = r, transform = ax.transAxes)
    ax.scatter(y,x, s=r, linewidths = lw, c = fc, edgecolors = ec, clip_on = False, zorder = 7)#, markersize = ms, markeredgewidth = lw, markerfacecolor = fc, markeredgecolor = ec)
    #ax.add_patch(circle)