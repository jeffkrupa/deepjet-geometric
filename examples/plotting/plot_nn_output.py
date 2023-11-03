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
import subprocess
sns.set_context("paper")
import mplhep as hep
import json
matplotlib.use('agg')
import tqdm
plt.style.use(hep.style.CMS)
import argparse
from matplotlib import gridspec
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

parser = argparse.ArgumentParser()
parser.add_argument('--ipath', action='store', type=str, help="Path to h5 files")
parser.add_argument('--is_n2', action='store_true', default=False, help="Plotting N2")
parser.add_argument('--which_qcd', action='store', type=str,help="All or specific qcd jet type 1..3, 5..8")
parser.add_argument('--copy', action='store_true', default=False,help="Copy files")

args = parser.parse_args()

training = args.ipath 
is_n2 = args.is_n2
which_qcd = args.which_qcd
if training.endswith('//'):
    print('no slash at the end')
    sys.exit()

def copy_files_locally(path):

    opath = path.split("/")[-2]
    os.system(f"mkdir -p local_file_storage/{opath}")
    abbreviated_path = path.replace("root://xrootd.cmsaf.mit.edu:1094","")
    files = subprocess.Popen([f"xrdfs root://xrootd.cmsaf.mit.edu:1094 ls {abbreviated_path}"],stdout=subprocess.PIPE, shell=True).communicate()[0].decode("utf-8")
    for f in str(files).split("\n"):

        os.system(f"xrdcp root://xrootd.cmsaf.mit.edu:1094/{f} ./local_file_storage/{opath}/")

    return f"local_file_storage/{opath}/"

if training.startswith("davs://") or training.startswith("root://"):
    if args.copy:
        training = copy_files_locally(training)
    else:
        training = "local_file_storage/"+training.split("/")[-2]+"/"


#training = sys.argv[1]
#is_n2 = int(sys.argv[2])
#which_qcd = sys.argv[3]

print("training", training)
print("is_n2",is_n2)
print("which_qcd",which_qcd)
if not any(which_qcd in ele for ele in ["all","1","2","3","5","6","7","8",]):
  print("specify which_qcd")
  sys.exit()
whichfeat = 999
if is_n2:
    whichfeat = 'n2'

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
    global training
    arr = None
    counter = 0

    pattern = "%s/%s_%s*h5"%(training,process,variation)
    print(pattern)
    for i in tqdm.tqdm(glob.glob(pattern)):
        counter += 1

        #if counter > 5:
        #    break
        print(i)        
        try:
            with h5py.File(i,'r') as f:
                if variable == 'n2':
                    feat = f['jet_kinematics'][()][:,-1]
                else:
                    feat = f['jet_features'][()]
                #print("hello",feat.shape)
                if "qcd" in process and which_qcd != "all":
                    feat = np.expand_dims(feat,axis=-1)
                    tmp_jettype = f['jet_type'][()]
                    feat = feat[tmp_jettype == int(which_qcd)]
                    #print("feat",feat)
                if arr is None:
                    arr = feat
                else:
                    arr = np.concatenate((arr,feat))
                print(feat.shape)
        except:
            pass
    print("arr",arr.shape)
    return arr

hist_dict = {}




fig = plt.figure()
gs = gridspec.GridSpec(7, 1, height_ratios=[1.8,0.5,0.5,0.5,0.5,0.5,0.5,])

binedges_global = [0,1,30]
if is_n2:
    binedges_global = [0.02,0.48,30]

def plot(axis,process,variation,variable,binedges,color,label,show=True):

    #print("In plotting function")
    arr = read_files(process,variation,variable,)


    tmpdict = {'val':np.squeeze(arr)}
    tmpdf = pd.DataFrame.from_dict(tmpdict)
    #print(tmpdf)
    oname=f"./../samples_inferred/{training.split('/')[-2]}/{process}_{variation}_{training.split('/')[-2]}_{which_qcd}.csv"
    
    if is_n2:
        os.system(f"mkdir -p ./../samples_inferred/{training.split('/')[-2]}_n2/") 
        oname=f"./../samples_inferred/{training.split('/')[-2]}_n2/{process}_{variation}_{training.split('/')[-2]}_n2_{which_qcd}.csv"
    #print(f"./../samples_inferred/{training.split('/')[-2]}/{process}_{variation}_nnout_{training.split('/')[-2]}_{variable}.csv")
    #tmpdf.to_csv(f"./../samples_inferred/{training.split('/')[-2]}/{process}_{variation}_nnout_{training.split('/')[-2]}_{variable}.csv")
    print(oname)
    tmpdf.to_csv(oname)

    
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
            #print(hist_dict[f'{str(variable)}_{process1}_{variation1}'][0][i])
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
ax_ratio5 = plt.subplot(gs[5])
ax_ratio6 = plt.subplot(gs[6])
ax.xaxis.set_zorder(99) 
ax.set_yscale('log')

#y_qcd, x_qcd, dummy = ax.hist(nom_jt_qcd[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
#y_qcd_isrUp, x_qcd_isrUp, dummy = ax.hist(isrUp_jt_qcd[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
#y_qcd_isrDown, x_qcd_isrDown, dummy = ax.hist(isrDown_jt_qcd[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
#y_higgs, x_higgs, dummy = ax.hist(nom_jt_higgs[:,-1],bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)

#plot_binned_data(ax, bins, y_qcd, histtype="step",stacked=False,color='salmon',label='QCD',linewidth=1.3,rwidth=2)
#plot_binned_data(ax, bins, y_higgs, histtype="step",stacked=False,color='midnightblue',label='Higgs',linewidth=1.3,rwidth=2)

qcd_label = {
  "all" : "QCD",
  "1" : "q",
  "2" : "c",
  "3" : "b",
  "5" : "g(qq)",
  "6" : "g(cc)",
  "7" : "g(bb)",
  "8" : "g(gg)",
}
qcd_legend_label = "QCD"
if which_qcd != 'all':
    qcd_legend_label += " [{qcd_label[which_qcd]}]"
plot(ax,'qcd','nominal',whichfeat,[0.05,0.5,30],'steelblue',f'QCD [{qcd_label[which_qcd]}]')
plot(ax,'higgs','nominal',whichfeat,[0.05,0.5,30],'magenta','Higgs')
plot(ax,'qcd','seed',whichfeat,[0.05,0.5,30],'yellow','seed',False)
plot(ax,'higgs','seed',whichfeat,[0.05,0.5,30],'yellow','seed',False)
plot(ax,'qcd','herwig',whichfeat,[0.05,0.5,30],'yellow','herwig',False)
plot(ax,'qcd','fsrRenHi',whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
plot(ax,'qcd','fsrRenLo',whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)
plot(ax,'higgs','herwig',whichfeat,[0.05,0.5,30],'yellow','herwig',False)
plot(ax,'higgs','fsrRenHi',whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
plot(ax,'higgs','fsrRenLo',whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)
#print(hist_dict.keys())
#print(binedges_global)
#bins = np.linspace(binedges_global[0],binedges_global[1],30)
label = "NN output"
if is_n2:
    label = "$N_2$"
ax_ratio6.set_xlabel(label)
ax.set_ylabel("Norm. to unit area",fontsize=21)
ax.legend(loc=(0.4,0.2))

#plot_ratio(ax_ratio1,'qcd','nominal','qcd','fsrRenHi',whichfeat,[0.05,0.5,30],'salmon',"$\mu(FSR)$ [QCD]",doboth=True,other='fsrRenLo')
#plot_ratio(ax_ratio2,'higgs','nominal','higgs','fsrRenHi',whichfeat,[0.05,0.5,30],'steelblue',"$\mu(FSR)$ [Higgs]",doboth=True,other='fsrRenLo')
#plot_ratio(ax_ratio3,'qcd','nominal','qcd','herwig',whichfeat,[0.05,0.5,30],'salmon',"Herwig7 [QCD]")
#plot_ratio(ax_ratio4,'higgs','nominal','higgs','herwig',whichfeat,[0.05,0.5,30],'steelblue',"Herwig7 [Higgs]")
NBINS=30


plot_ratio(ax_ratio1,'qcd','nominal','qcd','seed',whichfeat,[0.05,0.5,NBINS],'springgreen',f"seed [{qcd_label[which_qcd]}]")
plot_ratio(ax_ratio2,'higgs','nominal','higgs','seed',whichfeat,[0.05,0.5,NBINS],'indigo',f"seed [H]",)
plot_ratio(ax_ratio3,'qcd','nominal','qcd','fsrRenHi',whichfeat,[0.05,0.5,NBINS],'springgreen',f"FSR [{qcd_label[which_qcd]}]",doboth=True,other='fsrRenLo')
plot_ratio(ax_ratio4,'higgs','nominal','higgs','fsrRenHi',whichfeat,[0.05,0.5,NBINS],'indigo',f"FSR [H]",doboth=True,other='fsrRenLo')
plot_ratio(ax_ratio5,'qcd','nominal','qcd','herwig',whichfeat,[0.05,0.5,NBINS],'springgreen',f"Herwig7 [{qcd_label[which_qcd]}]")
plot_ratio(ax_ratio6,'higgs','nominal','higgs','herwig',whichfeat,[0.05,0.5,NBINS],'indigo',f"Herwig7 [H]")
axxes = [ax_ratio1,ax_ratio2,ax_ratio3,ax_ratio4,ax_ratio5,ax_ratio6]

#print(hist_dict)

fig.subplots_adjust(hspace=0.00)

ax.set_xlim([binedges_global[0],binedges_global[1]])
ax.xaxis.set_ticklabels([])

#axxes = [ax_ratio1,ax_ratio2,ax_ratio3,ax_ratio4]
for axx in axxes:
    if axx == axxes[0]:
        axx.set_ylabel("Variation/Nominal",fontsize=13)
    if axx != axxes[-1]:
        axx.xaxis.set_ticklabels([])
    if axx == axxes[-1] or axx == axxes[-2]:
        axx.set_ylim([0.,2.])
    else:
        axx.set_ylim([0.9,1.1])
    axx.set_xlim([binedges_global[0],binedges_global[1]]) 
    #axx.yaxis.set_ticklabels(['0.5','1.0','1.5'],fontsize=12)
    axx.tick_params(axis='y', which='major', labelsize=12)
    axx.axhline(y=1, linestyle='dashed',color='k')

label = "nn_output"
if is_n2:
    label = "n2_output"
if which_qcd != "all":
    label = label+"_"+which_qcd

print(f"/home/submit/jkrupa/public_html/cl/{training.split('/')[-2]}/{label}.png")
os.system(f"mkdir -p /home/submit/jkrupa/public_html/cl/{training.split('/')[-2]}/") 
plt.savefig(f"/home/submit/jkrupa/public_html/cl/{training.split('/')[-2]}/{label}.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/home/submit/jkrupa/public_html/cl/{training.split('/')[-2]}/{label}.pdf",dpi=300,bbox_inches='tight')


