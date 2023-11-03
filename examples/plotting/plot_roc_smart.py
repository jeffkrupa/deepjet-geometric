import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplhep as hep
import os,sys
from sklearn.metrics import auc
plt.style.use(hep.style.CMS)
import matplotlib
import seaborn
matplotlib.use('agg')
which_bkg_effs = [0.01,0.05,0.1,0.20]
which_sig_effs = [0.7,0.5]
n2_bins = np.linspace(0,0.6,30000)
nn_bins = np.concatenate((np.linspace(-0.001,0.00001,5000),np.linspace(0.00001,0.99999,1000),np.linspace(0.99999,1.0001,5000)))
name = sys.argv[1]
which_qcd = sys.argv[2]
if not any(which_qcd in ele for ele in ["all","1","2","3","5","6","7","8",]):
  print("specify which_qcd")
  sys.exit()
draw_arrows = False
do_n2=False
#variables = ['D_graph_jeff_FT_fixed_weights_5e6-testdataset', 'E_graph_jeff_FT_unfixed_weights_5e6-testdataset', 'F_graph_jeff_BCE_5e6-testdataset']#, 'F_graph_jeff_BCE_5e6-testdataset']
variables = ['graph_p_h_full_supervised_9_15-allaugs-RUN0', 'graph_p_h_full_supervised_9_15-allaugs-RUN1', 'graph_p_h_full_supervised_9_15-allaugs-RUN2', 'graph_p_h_full_supervised_9_15-allaugs-RUN4', 'graph_p_h_full_supervised_9_15-allaugs-RUN5','Graph_p_h_pretrained_unfixed-allaugs-RUN0','Graph_p_h_pretrained_unfixed-allaugs-RUN1','Graph_p_h_pretrained_unfixed-allaugs-RUN2','Graph_p_h_pretrained_unfixed-allaugs-RUN3','Graph_p_h_pretrained_unfixed-allaugs-RUN4','Graph_p_h_pretrained_unfixed-allaugs-5e5',] 
#variables = ['R_b_graph_jeff_FT_unfixed_weights_5e6-testdataset-run1', 'R_b_graph_jeff_FT_unfixed_weights_5e6-testdataset-run2', 'R_b_graph_jeff_FT_unfixed_weights_5e6-testdataset-run3', 'R_b_graph_jeff_FT_unfixed_weights_5e6-testdataset-run4', 'R_b_graph_jeff_FT_unfixed_weights_5e6-testdataset-run5','A_graph_jeff_FT_fixed_weights_5e6-testdataset','C_graph_jeff_BCE_5e6-testdataset',]
#variables = ['A_graph_jeff_FT_fixed_weights_5e6-testdataset', 'R_b_graph_jeff_FT_unfixed_weights_5e6-testdataset-run1', 'R_b_graph_jeff_FT_unfixed_weights_5e6-testdataset-run2','C_graph_jeff_BCE_5e6-testdataset']
reverse_order = [False]*len(variables)
#reverse_order = [False, False,False,False,False,False,False]
show_pythia = []
show_herwig = []
grouping = [0,0,0,0,0,1,1,1,1,1,None]
group_titles = { 0 : 'Fully-supervised', 1 : 'RS3L fine-tuned'}
labels = ['RS3L fully-supervised']*5 + ['RS3L fine-tuned']*5 + ['RS3L fine-tuned 5e5']
unc_labels = {'graph_p_h_full_supervised_9_15-allaugs' : 'Fully-supervised', 'Graph_p_h_pretrained_unfixed-allaugs' : 'RS3L fine-tuned', 'Graph_p_h_pretrained_unfixed-allaugs-5e5' : 'RS3L fine-tuned 5e5'}

print(labels)
print(variables)
bins = [nn_bins]*len(variables)
colors = ['grey','steelblue','fuchsia','limegreen','lightslategrey','grey','blue']*2
group_colors = {0 : "grey", 1 : "indianred",}
#group_shade_colors = {0 : "darkblue"}
tprs = {}
fprs = {}

arrow_dict = {}
arrow_dict_sig_eff = {}
#../samples_inferred/test_8doutput_merged5_withHiggs/qcd_nominal_nnout_test_8doutput_merged5_withHiggs.csv
def get_tpr_fpr(bins,variable,ts,fs,reverse=False):
    for v in ['nominal','fsrRenHi','fsrRenLo','herwig']:
        print(f'{v}_{variable}')
        counter = 0
        higgs_name = f"../samples_inferred/{variable}/higgs_{v}_{variable}.csv"
        qcd_name = f"../samples_inferred/{variable}/qcd_{v}_{variable}.csv"
        #if which_qcd != "all":
        qcd_name = qcd_name.replace(".csv",f"_{which_qcd}.csv")        
        higgs_name = higgs_name.replace(".csv",f"_{which_qcd}.csv") 
        print(qcd_name,higgs_name)       
        df_higgs = np.array(pd.read_csv(higgs_name)['val'].values.tolist())
        df_qcd =   np.array(pd.read_csv(qcd_name)['val'].values.tolist())
        #print(df_higgs)
        for i in bins:
            if not reverse:
                tpr = (df_higgs >= i).sum()/len(df_higgs)
                fpr = (df_qcd >= i).sum()/len(df_qcd)
            else:
                tpr = (df_higgs <= i).sum()/len(df_higgs)
                fpr = (df_qcd <= i).sum()/len(df_qcd)
            if '%s_%s'%(variable,v) not in tprs:
                tprs['%s_%s'%(variable,v)] = []
                tprs['%s_%s'%(variable,v)].append(tpr)
                fprs['%s_%s'%(variable,v)] = []
                fprs['%s_%s'%(variable,v)].append(fpr)
            else:
                tprs['%s_%s'%(variable,v)].append(tpr)
                fprs['%s_%s'%(variable,v)].append(fpr)

            if v == 'nominal':
                counter += 1
                if counter < 2:
                    continue
                for bkg_eff in which_bkg_effs:
                    if reverse:
                        if fprs['%s_%s'%(variable,v)][-1] > bkg_eff and fprs['%s_%s'%(variable,v)][-2] <= bkg_eff:
                           for v2 in ['nominal','fsrRenHi','fsrRenLo','herwig']:
                               higgs_name = f"../samples_inferred/{variable}/higgs_{v2}_{variable}.csv"
                               qcd_name = f"../samples_inferred/{variable}/qcd_{v2}_{variable}.csv"
                               #if which_qcd != "all":
                               qcd_name = qcd_name.replace(".csv",f"_{which_qcd}.csv")
                               higgs_name = higgs_name.replace(".csv",f"_{which_qcd}.csv")
                               tmp_higgs = np.array(pd.read_csv(higgs_name)['val'].values.tolist())                           
                               tmp_qcd = np.array(pd.read_csv(qcd_name)['val'].values.tolist())
                               arrow_dict['qcd_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_qcd <= i).sum()/len(tmp_qcd)
                               arrow_dict['higgs_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_higgs <= i).sum()/len(tmp_higgs)
                    else:
                        if fprs['%s_%s'%(variable,v)][-1] <= bkg_eff and fprs['%s_%s'%(variable,v)][-2] > bkg_eff:
                           for v2 in ['nominal','fsrRenHi','fsrRenLo','herwig']:
                               higgs_name = f"../samples_inferred/{variable}/higgs_{v2}_{variable}.csv"
                               qcd_name = f"../samples_inferred/{variable}/qcd_{v2}_{variable}.csv"
                               #if which_qcd != "all":
                               qcd_name = qcd_name.replace(".csv",f"_{which_qcd}.csv")
                               higgs_name = higgs_name.replace(".csv",f"_{which_qcd}.csv")
                           
                               tmp_higgs = np.array(pd.read_csv(higgs_name)['val'].values.tolist())
                               tmp_qcd = np.array(pd.read_csv(qcd_name)['val'].values.tolist())
                               arrow_dict['qcd_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_qcd >= i).sum()/len(tmp_qcd)
                               arrow_dict['higgs_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_higgs >= i).sum()/len(tmp_higgs)

                for sig_eff in which_sig_effs:
                    if reverse:
                        if tprs['%s_%s'%(variable,v)][-1] > sig_eff and tprs['%s_%s'%(variable,v)][-2] <= sig_eff:
                           for v2 in ['nominal','fsrRenHi','fsrRenLo','herwig']:
                               higgs_name = f"../samples_inferred/{variable}/higgs_{v2}_{variable}.csv"
                               qcd_name = f"../samples_inferred/{variable}/qcd_{v2}_{variable}.csv"
                               #if which_qcd != "all":
                               qcd_name = qcd_name.replace(".csv",f"_{which_qcd}.csv")
                               higgs_name = higgs_name.replace(".csv",f"_{which_qcd}.csv")
                               tmp_higgs = np.array(pd.read_csv(higgs_name)['val'].values.tolist())                           
                               tmp_qcd = np.array(pd.read_csv(qcd_name)['val'].values.tolist())
                               arrow_dict_sig_eff['qcd_%s_%s_eff_%s'%(v2,variable,str(sig_eff))] = (tmp_qcd <= i).sum()/len(tmp_qcd)
                               arrow_dict_sig_eff['higgs_%s_%s_eff_%s'%(v2,variable,str(sig_eff))] = (tmp_higgs <= i).sum()/len(tmp_higgs)
                    else:
                        if tprs['%s_%s'%(variable,v)][-1] <= sig_eff and tprs['%s_%s'%(variable,v)][-2] > sig_eff:
                           for v2 in ['nominal','fsrRenHi','fsrRenLo','herwig']:
                               higgs_name = f"../samples_inferred/{variable}/higgs_{v2}_{variable}.csv"
                               qcd_name = f"../samples_inferred/{variable}/qcd_{v2}_{variable}.csv"
                               #if which_qcd != "all":
                               qcd_name = qcd_name.replace(".csv",f"_{which_qcd}.csv")
                               higgs_name = higgs_name.replace(".csv",f"_{which_qcd}.csv")
                           
                               tmp_higgs = np.array(pd.read_csv(higgs_name)['val'].values.tolist())
                               tmp_qcd = np.array(pd.read_csv(qcd_name)['val'].values.tolist())
                               arrow_dict_sig_eff['qcd_%s_%s_eff_%s'%(v2,variable,str(sig_eff))] = (tmp_qcd >= i).sum()/len(tmp_qcd)
                               arrow_dict_sig_eff['higgs_%s_%s_eff_%s'%(v2,variable,str(sig_eff))] = (tmp_higgs >= i).sum()/len(tmp_higgs)

for i,var in enumerate(variables):
    get_tpr_fpr(bins[i],var,tprs,fprs,reverse_order[i])
#higgs_fsrRenHi_nnout_aug16_simclr_t0.1_fullData_epoch7_eff_0.05

def print_table(process,arrow_dict,title="",effs=[]):
 import csv, sys
 eff_dict = {}

 orig_stdout = sys.stdout
 oname = f'../samples_inferred/{name}/{process}_{title}_{which_qcd}.txt'
 os.system(f"mkdir ../samples_inferred/{name}/")
 f = open(oname, 'w')
 sys.stdout = f
 print("=====")
 print("PROCESS "+process)

 print(" ",end="\n") 
 for In, n in enumerate(variables):
   if 'n2' in n:
      print("N2")
   else:
      print("NN",labels[In])
   print("\t\tnom.\tfsrHi\tfsrLo\therwig",end="\n")
   for e in effs:
      print("\neff="+str(round(float(e)*100,0))+"%", end="\t")
      print(round(arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)],4),end="\t")
      eff_dict[f"{n}$nominal${e}"] = round(arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)],4)      
      for v in ["fsrRenHi","fsrRenLo","herwig"]:
         var = round(100.*((arrow_dict["%s_%s_%s_eff_%s"%(process,v,n,e)]-arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)]))/(arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)]),3,)
         print(var, end="\t") 
         eff_dict[f"{n}${v}${e}"] = var 
         #print("\t\t%s:"%v,round(100.*((arrow_dict["%s_%s_%s_eff_%s"%(process,v,n,e)]-arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)]))/(arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)]),2))
   print(" ",end="\n")
 sys.stdout = orig_stdout 
 f.close()
 print(eff_dict)

 def draw_uncs(eff):
   import collections
   import matplotlib.patches as mpatches
   sorted_eff_dict = []
   unique_trainingnames = []
   labels = []  
   sorted_eff_dict.append(collections.defaultdict(list))
   sorted_eff_dict.append(collections.defaultdict(list))
   sorted_eff_dict.append(collections.defaultdict(list))
   sorted_eff_dict.append(collections.defaultdict(list))
   for k,v in eff_dict.items():
     if eff not in k: 

       continue
     trainingname = k.split("$")[0]
     variation = k.split("$")[1]
     efficiency = k.split("$")[2]
     if trainingname not in unique_trainingnames: unique_trainingnames.append(trainingname)
     field_names = ["nominal","fsrRenHi","fsrRenLo","herwig"]
     if "RUN" in trainingname:
       trainingname = trainingname.split("-RUN")[0]
     print(trainingname,variation,v)

     sorted_eff_dict[field_names.index(variation)][trainingname].append(v)
   sorted_eff_dict = pd.DataFrame.from_dict(sorted_eff_dict)
   #for unique_name in unique_trainingnames:
   #    sorted_eff_dict = sorted_eff_dict.explode(unique_name)
   
   def add_label(violin, label):
      color = violin["bodies"][0].get_facecolor().flatten()
      labels.append((mpatches.Patch(color=color), label))

   fix,ax = plt.subplots()

   for col in sorted_eff_dict.columns:
       print(type(sorted_eff_dict[col])) 
       print(sorted_eff_dict[col].array)
       add_label(ax.violinplot(sorted_eff_dict[col].array[1:],positions=[1,2,3]),label=unc_labels[col]+f" {process} eff = {round(np.average(sorted_eff_dict[col].array[0]),4)}")
   plt.xticks(ticks=(1,2,3),labels=["fsrRenHi","fsrRenLo","herwig"], )#rotation="45")
   ax.set_ylabel("percent variation w.r.t. nominal")
   ax.set_xlabel("Systematic uncertainty")
   ax.legend(*zip(*labels), loc="upper left")
   text_to_add = "QCD eff"
   if "sig" in title:
      text_to_add = "H eff"
   ax.text(0.1,0.7,text_to_add + f" {int(100*float(eff))}%", transform=ax.transAxes)    
   plt.savefig(f"/home/submit/jkrupa/public_html/cl/{name}/uncertainties_{process}_{title}_{eff}.png",bbox_inches='tight')
 if "bkg" in title:  
     draw_uncs("0.01")
     draw_uncs("0.05")
     draw_uncs("0.1")
     draw_uncs("0.2")
 elif "sig" in title:
     draw_uncs("0.5")
     draw_uncs("0.7")



print_table("higgs",arrow_dict,title="bkg_eff",effs=[str(p) for p in which_bkg_effs])
print_table("qcd",arrow_dict,title="bkg_eff",effs=[str(p) for p in which_bkg_effs])
print_table("higgs",arrow_dict_sig_eff,title="sig_eff",effs=[str(p) for p in which_sig_effs])
print_table("qcd",arrow_dict_sig_eff,title="sig_eff",effs=[str(p) for p in which_sig_effs])

fig,ax = plt.subplots(figsize=(9,8))
ax.set_yscale('log')
groups = []
groups_tprs = []
#plt.plot(np.linspace(0.,1.,1000),np.linspace(0.,1.,1000),color='gold',label='random')

groups_done = [] #np.array()
for k in grouping:
    if k is not None and not k in groups_done:
        groups.append( [] )
        groups_tprs.append( [] )
        groups_done.append(k)
for i,var in enumerate(variables):
    for v in ['nominal',]:
        if grouping[i] is not None:
            groups[grouping[i]].append( fprs['%s_%s'%(var,v)] )
            groups_tprs[grouping[i]] =  tprs['%s_%s'%(var,v)] 
'''
new_variables = []
for i,var in enumerate(variables):
    if grouping[i] is None:

        new_variables.append(var)
variables = new_variables
'''
mins = {}
maxs = {}
means = {}
for iarray,array in enumerate(groups):
    mins[iarray] = np.array(array).min(axis=0)
    maxs[iarray] = np.array(array).max(axis=0)
    means[iarray] = np.array(array).mean(axis=0)

#print(len(groups))
#print("mins",mins)
#print("mins",maxs)
pairs = [] 
for i in range(len(mins)):
    pairs.append((mins[i],maxs[i]))
#sys.exit(1)
for i,var in enumerate(variables):
    for v in ['nominal','fsrRenHi','fsrRenLo','herwig']:
        if v == 'nominal':
            linestyle = 'solid'
        if 'fsrRen' in v:
            linestyle = 'dashed'
        if v == 'herwig':
            linestyle = 'dotted'
        label = labels[i] if v == 'nominal' else v
        if v != 'nominal':
            continue
        if grouping[i] is not None:
            continue
        plt.plot(tprs['%s_%s'%(var,v)],fprs['%s_%s'%(var,v)],
                 label=label + " AUC=%.4f"%(1.-auc(tprs['%s_%s'%(var,v)],fprs['%s_%s'%(var,v)])),color=colors[i],linestyle=linestyle)

for i,(minimum,maximum) in enumerate(pairs):
    for v in ['nominal',]:
        plt.fill_between(groups_tprs[i], minimum, maximum, 
                 alpha=0.3, color=group_colors[i], )#+ " AUC=%.4f"%(1.-auc(tprs['%s_%s'%(var,v)],fprs['%s_%s'%(var,v)])),color=colors[i],linestyle=linestyle)
        plt.plot(groups_tprs[i],means[i], label = group_titles[i]+" AUC=%.4f"%(1-auc(groups_tprs[i],means[i])),color= group_colors[i])


for i,var in enumerate(variables):
    if draw_arrows == False:
        continue
    for v in ['fsrRenHi','fsrRenLo','herwig']:
        for bkg_eff in which_bkg_effs:
            plt.plot([arrow_dict['higgs_nominal_%s_eff_%s'%(var,str(bkg_eff))],arrow_dict['higgs_%s_%s_eff_%s'%(v,var,str(bkg_eff))]],\
                     [arrow_dict['qcd_nominal_%s_eff_%s'%(var,str(bkg_eff))],arrow_dict['qcd_%s_%s_eff_%s'%(v,var,str(bkg_eff))]],color=colors[i])
            #ax.arrow(arrow_dict['higgs_nominal_%s_eff_%s'%(var,str(bkg_eff))], \
            #arrow_dict['qcd_nominal_%s_eff_%s'%(var,str(bkg_eff))], (arrow_dict['higgs_%s_%s_eff_%s'%(v,var,str(bkg_eff))]-arrow_dict['higgs_nominal_%s_eff_%s'%(var,str(bkg_eff))]), \
            #(arrow_dict['qcd_%s_%s_eff_%s'%(v,var,str(bkg_eff))]-arrow_dict['qcd_nominal_%s_eff_%s'%(var,str(bkg_eff))]), fc=colors[i], ec=colors[i]) 

#get_tpr_fpr(nn_bins,'n2',tprs,fprs)
ax.set_xlabel("Higgs acceptance",fontsize=24)
if which_qcd != "all":
    label = label+"_"+which_qcd
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
    qcd_legend_label += f" [{qcd_label[which_qcd]}]"

ax.set_ylabel(f"{qcd_legend_label} fake rate",fontsize=24)
ax.set_yscale('log')
plt.grid(which='both')
plt.legend(fontsize=13)
ax.set_ylim([0.0002,.08])
os.system(f"mkdir -p /home/submit/jkrupa/public_html/cl/{name}/")
plt.savefig(f"/home/submit/jkrupa/public_html/cl/{name}/roc_test_{which_qcd}.pdf",bbox_inches='tight')
plt.savefig(f"/home/submit/jkrupa/public_html/cl/{name}/roc_test_{which_qcd}.png",bbox_inches='tight',dpi=300)

ax.set_xscale("log")
#ax.set_xlim([0.0005,2])
ax.set_xlim([0.2,1.])
plt.ticklabel_format(axis="y", style="plain",)
plt.savefig(f"/home/submit/jkrupa/public_html/cl/{name}/roc_test_{which_qcd}_logx.pdf",bbox_inches='tight')
plt.savefig(f"/home/submit/jkrupa/public_html/cl/{name}/roc_test_{which_qcd}_logx.png",bbox_inches='tight',dpi=300)
