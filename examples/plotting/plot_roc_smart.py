import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplhep as hep
import os,sys
plt.style.use(hep.style.CMS)

which_bkg_effs = [0.01,0.05,0.1,0.2]
n2_bins = np.linspace(0,0.6,300)
nn_bins = np.concatenate((np.linspace(-0.001,0.9994,10000),np.linspace(0.9994,1.00,10000)))
name = sys.argv[1]
draw_arrows = False

#variables = ['n2','nnout_t0p1','nnout_t0p1_dim512', 'nnout_vicreg_dim8','nnout_supervised']
variables = ['nnout_aug16_simclr_t0.1_fullData_epoch7','nnout_aug16_simclr_t0.1_fullData_epoch66']#'nnout_aug16_simclr_t0.05_fullData_epoch7','nnout_aug16_vicreg_mse1_var1_cov10_fullData_epoch11','nnout_aug16_simclr_t0.075_fullData_epoch14']
reverse_order = [False,False,False,False,False]
#labels = ['$N_2$','LUST ($\\tau=0.1$)','LUST ($\\tau=0.1$, dim512)','LUST (VicReg)','Fully supervised']
labels = ['LUST ($\\tau=0.1$ epoch7)','LUST ($\\tau=0.1$ epoch66)','VICREG (mse=1,var=1,cov=10)', 'LUST ($\\tau=0.075$)']
#bins = [n2_bins,nn_bins,nn_bins,nn_bins,nn_bins,nn_bins]
bins = [nn_bins,nn_bins,nn_bins,nn_bins]
colors = ['indianred','steelblue','fuchsia','limegreen','lightslategrey']
tprs = {}
fprs = {}

arrow_dict = {}

def get_tpr_fpr(bins,variable,ts,fs,reverse=False):
    for v in ['nominal','fsrRenHi','fsrRenLo','herwig']:
        print(f'{v}_{variable}')
        counter = 0
        df_higgs = np.array(pd.read_csv(f"higgs_{v}_{variable}.csv")['val'].values.tolist())
        df_qcd =   np.array(pd.read_csv(f"qcd_{v}_{variable}.csv")['val'].values.tolist())
        print(df_higgs)
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
                               tmp_higgs = np.array(pd.read_csv(f"higgs_{v2}_{variable}.csv")['val'].values.tolist())                           
                               tmp_qcd = np.array(pd.read_csv(f"qcd_{v2}_{variable}.csv")['val'].values.tolist())
                               arrow_dict['qcd_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_qcd <= i).sum()/len(tmp_qcd)
                               arrow_dict['higgs_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_higgs <= i).sum()/len(tmp_higgs)
                    else:
                        if fprs['%s_%s'%(variable,v)][-1] <= bkg_eff and fprs['%s_%s'%(variable,v)][-2] > bkg_eff:
                           for v2 in ['nominal','fsrRenHi','fsrRenLo','herwig']:
                               tmp_higgs = np.array(pd.read_csv(f"higgs_{v2}_{variable}.csv")['val'].values.tolist())
                               tmp_qcd = np.array(pd.read_csv(f"qcd_{v2}_{variable}.csv")['val'].values.tolist())
                               arrow_dict['qcd_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_qcd >= i).sum()/len(tmp_qcd)
                               arrow_dict['higgs_%s_%s_eff_%s'%(v2,variable,str(bkg_eff))] = (tmp_higgs >= i).sum()/len(tmp_higgs)


for i,var in enumerate(variables):
    get_tpr_fpr(bins[i],var,tprs,fprs,reverse_order[i])

#higgs_fsrRenHi_nnout_aug16_simclr_t0.1_fullData_epoch7_eff_0.05

def print_table(process,arrow_dict):
 print("=====")
 print("PROCESS "+process)
 for n in variables:
   for e in ["0.2","0.1","0.05","0.01"]:
      print("\tEFFICIENCY "+e)
      for v in ["fsrRenHi","fsrRenLo","herwig"]:
         print("\t\t%s"%v,100.*((arrow_dict["%s_%s_%s_eff_%s"%(process,v,n,e)]-arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)]))/(arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)]))

      print("\t\tnominal: ",arrow_dict["%s_%s_%s_eff_%s"%(process,"nominal",n,e)])
print_table("higgs",arrow_dict)
print_table("qcd",arrow_dict)

print("arrow_dict",arrow_dict)

fig,ax = plt.subplots(figsize=(9,8))
ax.set_yscale('log')
plt.plot(np.linspace(0.,1.,1000),np.linspace(0.,1.,1000),color='gold',label='random')

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
        plt.plot(tprs['%s_%s'%(var,v)],fprs['%s_%s'%(var,v)],
                 label=label,color=colors[i],linestyle=linestyle)


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
ax.set_ylabel("QCD fake rate",fontsize=24)
ax.set_yscale('log')
plt.grid(which='both')
plt.legend(fontsize=9)
ax.set_ylim([0.002,2])

os.system(f"mkdir -p /home/tier3/jkrupa/public_html/cl/{name}/")
plt.savefig(f"/home/tier3/jkrupa/public_html/cl/{name}/roc_test.pdf",bbox_inches='tight')
plt.savefig(f"/home/tier3/jkrupa/public_html/cl/{name}/roc_test.png",bbox_inches='tight',dpi=300)


