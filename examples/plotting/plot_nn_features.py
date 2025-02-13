import sys
import os
import glob
import h5py
import numpy as np
import awkward as ak
import uproot as uproot
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import awkward as ak
sns.set_context("paper")
import mplhep as hep
import json
import tqdm
plt.style.use(hep.style.CMS)

from sklearn.preprocessing import normalize

from matplotlib import gridspec
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

feature_labels = { 
    0 : "NN feature 0", 
    1 : "NN feature 1", 
    2 : "NN feature 2", 
    3 : "NN feature 3", 
    4 : "NN feature 4", 
    5 : "NN feature 5", 
    6 : "NN feature 6", 
    7 : "NN feature 7", 
    #    8 : "NN feature 8", 
    #    9 : "NN feature 9", 
    #    10 : "NN feature 10", 
    #    11 : "NN feature 11", 
    #    12 : "NN feature 12", 
    #    13 : "NN feature 13", 
    #    14 : "NN feature 14", 
    #    15 : "NN feature 15", 
    #    16 : "NN feature 16", 
    #    17 : "NN feature 17", 
    #    18 : "NN feature 18", 
    #    19 : "NN feature 19", 
    #    20 : "NN feature 20", 
    #    21 : "NN feature 21", 
    #    22 : "NN feature 22", 
    #    23 : "NN feature 23", 
    #    24 : "NN feature 24", 
    #    25 : "NN feature 25", 
    #    26 : "NN feature 26", 
    #    27 : "NN feature 27", 
    #    28 : "NN feature 28", 
    #    29 : "NN feature 29", 
    #    30 : "NN feature 30", 
    #    31 : "NN feature 31", 
    32 : "$\mathrm{Jet\\ p_T}$",
    33 : "$\mathrm{Jet\\ \eta}$",
    34 : "$\mathrm{Jet\\ \phi}$",
    35 : "$\mathrm{Jet\\ m_{SD}}$",
    36 : "$\mathrm{Jet\\ N_2}$",
}
whichfeat = int(sys.argv[1])
whichtraining = sys.argv[2]
print("whichtraining", whichtraining.split('/')[-2])
def plot_binned_data(axes, binedges, data,
               *args, **kwargs):
    #The dataset values are the bin centres
    x = (binedges[1:] + binedges[:-1]) / 2.0
    #The weights are the y-values of the input binned data
    weights = data
    return axes.hist(x, bins=binedges, weights=weights,
               *args, **kwargs)




def fill_between_steps(ax, x, y1, y2=0, step_where='pre', **kwargs):
    ''' fill between a step plot and 

    Parameters
    ----------
    ax : Axes
       The axes to draw to

    x : array-like
        Array/vector of index values.

    y1 : array-like or float
        Array/vector of values to be filled under.
    y2 : array-Like or float, optional
        Array/vector or bottom values for filled area. Default is 0.

    step_where : {'pre', 'post', 'mid'}
        where the step happens, same meanings as for `step`

    **kwargs will be passed to the matplotlib fill_between() function.

    Returns
    -------
    ret : PolyCollection
       The added artist

    '''
    x_tmp=x
    x=x[:-1]
    
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays 
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array 

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    if step_where == 'pre':
        steps = np.ma.zeros((3, 2 * len(x) - 1), np.float)
        #print steps,vertices
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        #print steps,vertices
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]
        #print steps,vertices

    elif step_where == 'post':
        steps = np.ma.zeros((3, 2 * len(x) -1), np.float)
       
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]
        #print steps
        #print vertices

    elif step_where == 'mid':
        steps = np.ma.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
        #print steps#,vertices
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    xx, yy1, yy2 = steps
    #print xx
    #xx = np.append(xx, xx[len(xx)-1]+(xx[1]-xx[0]))
    xx = np.append(xx, xx[len(xx)-1]+(x_tmp[len(x_tmp)-1]-x_tmp[len(x_tmp)-2]))
    #print xx
    yy1 = np.append(yy1,yy1[len(yy1)-1])
    yy2 = np.append(yy2,yy2[len(yy2)-1])
    #xx.append(xx[len(xx)-1])
    #print xx,yy1
    # now to the plotting part:
    #print(**kwargs)
    return ax.fill_between(xx, yy1, y2=yy2,alpha=0.4,linewidth=0.0,**kwargs)
    #return ax.fill_between(xx, yy1, y2=yy2,alpha=0.3,color='cyan')

def useEnvelope(up, down):
    for i in range(up.size):
        if(up[i] < 1 and down[i] < 1):
            if (up[i] > down[i]):
                up[i] = 1
            else:
                down[i] = 1   
        
        if(up[i] > 1 and down[i] > 1):
            if (up[i] > down[i]):
                down[i] = 1
            else:
                up[i] = 1   



def read_files(process,variation,variable):
    arr = None
    #print("/mnt/T2_US_MIT/hadoop/cms/store/user/bmaier/cl/outfiles/july29_%s/%s_*%s*h5"%(whichtraining,process,variation))
    counter = 0
    #print(glob.glob(whichtraining+"/%s_*%s*h5"%(process,variation)))
    for i in tqdm.tqdm(glob.glob(whichtraining+"/%s_*%s*h5"%(process,variation))):
        counter += 1
        np.set_printoptions(threshold=sys.maxsize)
        #if counter > 1:
        #    break
        
        try:
            with h5py.File(i,'r') as f:
                feature_arr = f['jet_features'][()]
                #feature_arr = feature_arr[:,:8]
                #feature_arr[:,:len(feature_labels)-5] = normalize(feature_arr[:,:len(feature_labels)-5], axis=1)
                #print(feature_arr.shape)
                feat = feature_arr[:,variable]
                #feat = normalize(feat, axis=1)#np.amax(feature_arr[:,:16], axis=1)
                #print(feat)
                #print("alala")
                feat = feat[~np.isnan(feat)]
                #print("blblb")
                #if np.isnan(feat).any():
                    #print("WTFFFFF")
                    #print(feat)
                if arr is None:
                    arr = feat
                else:
                    arr = np.concatenate((arr,feat))
        except:
            pass
    arr
    return arr

hist_dict = {}



fig = plt.figure()
gs = gridspec.GridSpec(5, 1, height_ratios=[1.8,0.5,0.5,0.5,0.5])

binedges_global = -1

def plot(axis,process,variation,variable,binedges,color,label,show=True):

    #print("In plotting function")
    arr = read_files(process,variation,variable)

    global binedges_global

    if binedges_global == -1:
        #print("WTFFFFFF")
        binedges_global = [np.min(arr),np.max(arr),30]


    #if binedges_global[0] > np.min(arr):
    #    binedges_global[0] = np.min(arr)
    #if binedges_global[1] < np.max(arr):
    #    binedges_global[1] = np.max(arr)
        
    bins = np.linspace(binedges_global[0],binedges_global[1],binedges_global[2])
    #print(bins)
    y, x, dummy = axis.hist(arr,bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
    
    #print(y,x,dummy)
    if show:
        _ = plot_binned_data(axis, bins, y, histtype="step",stacked=False,color=color,label=label,linewidth=1.3,rwidth=2)
    else:
        _ = plot_binned_data(axis, bins, y, histtype="step",stacked=False,color=color,linewidth=1.3,rwidth=2,alpha=0.)
        
    hist_dict[f'{str(variable)}_{process}_{variation}'] = _


def plot_ratio(axis,process1,variation1,process2,variation2,variable,binedges,color,text=None,doboth=False,other=None):

    global binedges_global
    
    bins = np.linspace(binedges_global[0],binedges_global[1],binedges_global[2])

    #print(bins)
    #print(hist_dict[f'{variable}_{process1}_{variation1}'][0]/hist_dict[f'{variable}_{process2}_{variation2}'][0])
    y,x,dummy = plot_binned_data(axis, bins, np.nan_to_num(hist_dict[f'{str(variable)}_{process1}_{variation2}'][0]/hist_dict[f'{str(variable)}_{process2}_{variation1}'][0],copy=True,posinf=0), histtype="step",stacked=False,color=color,linewidth=0.0,rwidth=2)
    if doboth:

        y_other,x_other,dummy = plot_binned_data(axis, bins, np.nan_to_num(hist_dict[f'{str(variable)}_{process1}_{other}'][0]/hist_dict[f'{str(variable)}_{process2}_{variation1}'][0],copy=True,posinf=0), histtype="step",stacked=False,color=color,linewidth=0.0,rwidth=2)
        fill_between_steps(axis,bins, y, y_other, step_where="post",color=color,zorder=0)
    else:
        binentries = []
        for i in range(len(hist_dict[f'{str(variable)}_{process1}_{variation1}'][0])):
            print(hist_dict[f'{str(variable)}_{process1}_{variation1}'][0][i])
            if hist_dict[f'{str(variable)}_{process1}_{variation1}'][0][i] == 0:
                binentries.append(0)
            else:
                binentries.append(1)

        lower = []
        for i in range(len(hist_dict[f'{str(variable)}_{process1}_{variation1}'][0])):
            if y[i] < 1:
                lower.append(y[i]-0.05)
            else:
                lower.append(0.98)
        fill_between_steps(axis,bins, y, binentries , step_where="post",color=color,zorder=0)
        #print(lower)
        #plot_binned_data(axis, bins, lower, histtype="step",stacked=False,color='white',linewidth=1.6,rwidth=2)
    axis.text(0.03,0.7,text,transform=axis.transAxes,fontsize=14)


ax = plt.subplot(gs[0])
ax_ratio1 = plt.subplot(gs[1])
ax_ratio2 = plt.subplot(gs[2])
ax_ratio3 = plt.subplot(gs[3])
ax_ratio4 = plt.subplot(gs[4])
ax.xaxis.set_zorder(99) 
ax.set_yscale('log')

#y_qcd, x_qcd, dummy = ax.hist(nom_jt_qcd[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
#y_qcd_isrUp, x_qcd_isrUp, dummy = ax.hist(isrUp_jt_qcd[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
#y_qcd_isrDown, x_qcd_isrDown, dummy = ax.hist(isrDown_jt_qcd[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
#y_higgs, x_higgs, dummy = ax.hist(nom_jt_higgs[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)

#plot_binned_data(ax, bins, y_qcd, histtype="step",stacked=False,color='springgreen',label='QCD',linewidth=1.3,rwidth=2)
#plot_binned_data(ax, bins, y_higgs, histtype="step",stacked=False,color='midnightblue',label='Higgs',linewidth=1.3,rwidth=2)


plot(ax,'qcd','nominal',whichfeat,[0.05,0.5,30],'springgreen','QCD')
#plot(ax,'higgs','nominal',whichfeat,[0.05,0.5,30],'indigo','Higgs')

plot(ax,'qcd','herwig',whichfeat,[0.05,0.5,30],'yellow','herwig',False)
plot(ax,'qcd','fsrRenHi',whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
plot(ax,'qcd','fsrRenLo',whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)
#plot(ax,'higgs','herwig',whichfeat,[0.05,0.5,30],'yellow','herwig',False)
#plot(ax,'higgs','fsrRenHi',whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
#plot(ax,'higgs','fsrRenLo',whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)

print(binedges_global)
#bins = np.linspace(binedges_global[0],binedges_global[1],30)

ax_ratio4.set_xlabel(feature_labels[whichfeat])
ax.set_ylabel("Norm. to unit area",fontsize=21)
ax.legend(loc=(0.4,0.2))

plot_ratio(ax_ratio1,'qcd','nominal','qcd','fsrRenHi',whichfeat,[0.05,0.5,30],'springgreen',"$\mu(FSR)$ [QCD]",doboth=True,other='fsrRenLo')
plot_ratio(ax_ratio2,'higgs','nominal','higgs','fsrRenHi',whichfeat,[0.05,0.5,30],'indigo',"$\mu(FSR)$ [Higgs]",doboth=True,other='fsrRenLo')
plot_ratio(ax_ratio3,'qcd','nominal','qcd','herwig',whichfeat,[0.05,0.5,30],'springgreen',"Herwig7 [QCD]")
plot_ratio(ax_ratio4,'higgs','nominal','higgs','herwig',whichfeat,[0.05,0.5,30],'indigo',"Herwig7 [Higgs]")


#print(hist_dict)

fig.subplots_adjust(hspace=0.00)

ax.set_xlim([binedges_global[0],binedges_global[1]])
ax.xaxis.set_ticklabels([])

axxes = [ax_ratio1,ax_ratio2,ax_ratio3,ax_ratio4]
for axx in axxes:
    if axx == axxes[0]:
        axx.set_ylabel("Variation/Nominal",fontsize=13)
    if axx != axxes[-1]:
        axx.xaxis.set_ticklabels([])
    axx.set_xlim([binedges_global[0],binedges_global[1]])
    axx.set_ylim([0.4,1.6])
    #axx.yaxis.set_ticklabels(['0.5','1.0','1.5'],fontsize=12)
    axx.tick_params(axis='y', which='major', labelsize=12)
    axx.axhline(y=1, linestyle='dashed',color='k')
    
os.system(f"mkdir -p /home/tier3/jkrupa/public_html/cl/{whichtraining.split('/')[-2]}/")
plt.savefig("/home/tier3/jkrupa/public_html/cl/%s/feat_%s.png"%(whichtraining.split('/')[-2],str(whichfeat)),dpi=300,bbox_inches='tight')
plt.savefig("/home/tier3/jkrupa/public_html/cl/%s/feat_%s.pdf"%(whichtraining.split('/')[-2],str(whichfeat)),dpi=300,bbox_inches='tight')


