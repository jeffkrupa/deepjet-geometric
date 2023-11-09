import os.path as osp
import glob
import sys 
import os
import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class CLV2_Nate2(Dataset):
    r'''
        input x0: (PF candidates)

        # truth labels = jet class
    '''

    url = '/dummy/'

    def __init__(self, root, qcd_only=True, seed_only=False, herwig_only=False, which_augmentations=None, kinematics_only=False, abseta=False, opath="", dry_run=False, istransformer=False, transform=None, vartype_training=False,Nmaxsample = 2e3,gvq = False,args = None,add_SSL = False):
        super(CLV2_Nate2, self).__init__(root, transform)
        print(kinematics_only,opath,"opath")
        self.qcd_only = qcd_only
        self.seed_only = seed_only 
        self.herwig_only = herwig_only
        self.which_augmentations = which_augmentations
        self.kinematics_only = kinematics_only 
        self.abseta = abseta
        self.dry_run = dry_run
        self.istransformer = istransformer
        self.vartype_training = vartype_training
        self.Nmaxsample = Nmaxsample
        self.gvq = gvq
        self.args = args
        self.add_SSL = add_SSL
        self.fill_data()
        stdoutOrigin=sys.stdout
        
        sys.stdout = open("./"+opath+"_log.txt", "w")
        if args.top_zbb:
            print("# of zbb jets={}, {:.2f}%".format(np.sum(self.data_jettype==0), 100*np.sum(self.data_jettype==0)/len(self.data_jettype)))
            print("# of top jets={}, {:.2f}%".format(np.sum(self.data_jettype==1), 100*np.sum(self.data_jettype==1)/len(self.data_jettype)))
        else:
            print("# of q jets={}, {:.2f}%".format(np.sum(self.data_jettype==1), 100*np.sum(self.data_jettype==1)/len(self.data_jettype)))
            print("# of c jets={}, {:.2f}%".format(np.sum(self.data_jettype==2), 100*np.sum(self.data_jettype==2)/len(self.data_jettype)))
            print("# of b jets={}, {:.2f}%".format(np.sum(self.data_jettype==3), 100*np.sum(self.data_jettype==3)/len(self.data_jettype)))
            print("# of H jets={}, {:.2f}%".format(np.sum(self.data_jettype==4), 100*np.sum(self.data_jettype==4)/len(self.data_jettype)))
            print("# of qq jets={}, {:.2f}%".format(np.sum(self.data_jettype==5), 100*np.sum(self.data_jettype==5)/len(self.data_jettype)))
            print("# of cc jets={}, {:.2f}%".format(np.sum(self.data_jettype==6), 100*np.sum(self.data_jettype==6)/len(self.data_jettype)))
            print("# of bb jets={}, {:.2f}%".format(np.sum(self.data_jettype==7), 100*np.sum(self.data_jettype==7)/len(self.data_jettype)))
            print("# of gg jets={}, {:.2f}%".format(np.sum(self.data_jettype==8), 100*np.sum(self.data_jettype==8)/len(self.data_jettype)))
            print("# of Zbb jets={}, {:.2f}%".format(np.sum(self.data_jettype==9), 100*np.sum(self.data_jettype==9)/len(self.data_jettype)))
            print("# of Zqq jets={}, {:.2f}%".format(np.sum(self.data_jettype==10), 100*np.sum(self.data_jettype==10)/len(self.data_jettype)))
            print("# of top jets={}, {:.2f}%".format(np.sum(self.data_jettype==11), 100*np.sum(self.data_jettype==11)/len(self.data_jettype)))

        print("# of nominal jets={}, {:.2f}%".format(np.sum(self.data_vartype==-1), 100*np.sum(self.data_vartype==-1)/len(self.data_vartype)))
        print("# of seed jets={}, {:.2f}%".format(np.sum(self.data_vartype==0), 100*np.sum(self.data_vartype==0)/len(self.data_vartype)))
        print("# of fsrLo jets={}, {:.2f}%".format(np.sum(self.data_vartype==1), 100*np.sum(self.data_vartype==1)/len(self.data_vartype)))
        print("# of fsrHi jets={}, {:.2f}%".format(np.sum(self.data_vartype==2), 100*np.sum(self.data_vartype==2)/len(self.data_vartype)))
        print("# of herwig jets={}, {:.2f}%".format(np.sum(self.data_vartype==3), 100*np.sum(self.data_vartype==3)/len(self.data_vartype)))
        sys.stdout.close()
        sys.stdout=stdoutOrigin
   
    def fill_data(self):
        #add the files from /work/tier3/jkrupa/cl/samples/mar20/outfiles/contrastive_train
#         if self.add_SSL:
#             print(len(self.raw_paths))
#             p = '/work/tier3/jkrupa/cl/samples/mar20/outfiles/contrastive_train/raw'
#             tmp = []
#             for a in os.listdir(p):
#                 tmp.append(os.path.join(p,a))
#             tmp_2 = self.raw_paths
#             tmp_2.extend(tmp)
#             setattr(self, 'raw_paths', tmp_2)
#             print(len(self.raw_paths))
#             print(self.raw_paths)
            
#             print('Adding files from SSL training')
        print("Reading files...")
        
        for fi,path in enumerate(tqdm(self.raw_paths[:])):
            with h5py.File(path, 'r') as f:
                tmp_features = f['features'][()].astype(np.float32)
                #print(tmp_features.shape)
                if self.abseta:
                    tmp_features[:,:,2] = np.abs(tmp_features[:,:,2])
                tmp_truth_label = f['truth_label'][()]
                tmp_jettype = f['jettype'][()]
                tmp_vartype = f['vartype'][()]
                 
  

                 #### VARTYPE
                 #nominal = -1
                 #seed = 0
                 #fsrRenHi = 1
                 #fsrRenLo = 2
                 #herwig = 3

                 #### JETTYPE
                 #qcd_qlight 1
                 #qcd_qc 2
                 #qcd_qb 3
                 #Higgs 4
                 #qcd_glightlight 5
                 #qcd_gcc 6
                 #qcd_gbb 7
                 #qcd_ggg 8

                if self.vartype_training:
                    tmp_features = tmp_features[(tmp_vartype==-1) | (tmp_vartype==3)]
                    tmp_jettype = tmp_jettype[(tmp_vartype==-1) | (tmp_vartype==3)]
                    tmp_truth_label = tmp_truth_label[(tmp_vartype==-1) | (tmp_vartype==3)]
                    tmp_vartype = tmp_vartype[(tmp_vartype==-1) | (tmp_vartype==3)]
                if self.qcd_only:
                    tmp_features = tmp_features[tmp_truth_label==0]
                    tmp_vartype = tmp_vartype[tmp_truth_label==0]
                    tmp_jettype = tmp_jettype[tmp_truth_label==0]
                    tmp_truth_label = tmp_truth_label[tmp_truth_label==0]
                ''' 
                if self.which_augmentations is not None:
                    idxs_to_keep = np.zeros(len(tmp_features),dtype=int)
                    for augmentation in self.which_augmentations:
                        is_aug_only = np.argwhere(tmp_vartype==augmentation,)
                        idxs_to_keep[[is_aug_only]] = 1
                        idxs_to_keep[[[is_aug-1 for is_aug in is_aug_only]]] = 1
                    mask = idxs_to_keep.astype(bool)
                    tmp_features = tmp_features[mask,:]
                    tmp_vartype = tmp_vartype[mask]
                    tmp_jettype = tmp_jettype[mask]
                    tmp_truth_label = tmp_truth_label[mask,]
                '''
                  
                if self.gvq:
                    tmp_jettype[(tmp_jettype <=3) & (tmp_jettype >= 1)] = 0
                    tmp_jettype[(tmp_jettype <=8) & (tmp_jettype >= 5)] = 1
                    tmp_features= tmp_features[tmp_jettype != 4]
                    tmp_truth_label= tmp_truth_label[tmp_jettype != 4]
                    tmp_vartype= tmp_vartype[tmp_jettype != 4]
                    tmp_jettype = tmp_jettype[tmp_jettype != 4]
                elif self.args.zbb_qcd:
                    #For Zbb vs QCD, you want only jet_type==9 (set as signal, truthlabel=1) and jet type in {1,2,3,5,6,7,8}(set as bkg, truthlabel=0) to be in the training
                    tmp_jettype[(tmp_jettype !=4) & (tmp_jettype != 9)] = 0
                    tmp_jettype[(tmp_jettype ==9)] = 1
                    tmp_features= tmp_features[tmp_jettype != 4]
                    tmp_truth_label= tmp_truth_label[tmp_jettype != 4]
                    tmp_vartype= tmp_vartype[tmp_jettype != 4]
                    tmp_jettype = tmp_jettype[tmp_jettype != 4]
                elif self.args.top_zbb:
                    tmp_jettype[tmp_jettype == 9] = 0
                    tmp_jettype[tmp_jettype == 11] = 1

                if self.args.fine_tuning:
                     
                     #if ft_with_p_h, we filter out all vartypes that are not class 0 or 3
                     
                    if self.args.ft_with_p_h:
                      if not self.args.all_augmentations :
                        vartype = torch.tensor(tmp_vartype).float()
                        mask = (vartype == -1) | (vartype == 3)
                        tmp_vartype = tmp_vartype[mask]
                        tmp_truth_label = tmp_truth_label[mask]
                        tmp_features = tmp_features[mask]
                        tmp_jettype = tmp_jettype[mask]
                        
#                          if only fine_tuning passed, we fine tune on nominal only
                    elif self.args.ft_with_p:
                        tmp_vartype = tmp_vartype[0::2]
                        tmp_truth_label = tmp_truth_label[0::2]
                        tmp_features = tmp_features[0::2]
                        tmp_jettype = tmp_jettype[0::2]
                    else:
                        #print("we're here")
                        if self.which_augmentations is not None:
                            idxs_to_keep = np.zeros(len(tmp_features),dtype=int)
                            for augmentation in self.which_augmentations:
                                is_aug_only = np.argwhere(tmp_vartype==augmentation,)
                                idxs_to_keep[[is_aug_only]] = 1
                                idxs_to_keep[[[is_aug-1 for is_aug in is_aug_only]]] = 1
                            mask = idxs_to_keep.astype(bool)
                            tmp_features = tmp_features[mask,:]
                            tmp_vartype = tmp_vartype[mask]
                            tmp_jettype = tmp_jettype[mask]
                            tmp_truth_label = tmp_truth_label[mask,]

                        '''
                        vartype = torch.tensor(tmp_vartype).float()
                        mask = torch.zeros_like(vartype).bool()
                        mask[vartype == -1] = True
                        if self.args.add_reseeded:
                            #print("add_reseeded")
                            mask[vartype == 0] = True
                        if self.args.add_fsr:
                            #print("add_fsr")
                            mask[vartype == 1] = True 
                            mask[vartype == 2] = True
                        tmp_vartype = tmp_vartype[mask]
                        tmp_truth_label = tmp_truth_label[mask]
                        tmp_features = tmp_features[mask]
                        tmp_jettype = tmp_jettype[mask]
                        #print("mask",mask[:20])
                        #print("vartype",vartype[:20])
                        '''
                if fi == 0:
                    self.data_features = tmp_features
                    self.data_truth_label = tmp_truth_label
                    self.data_vartype = tmp_vartype
                    self.data_jettype = tmp_jettype
                else:
                    self.data_features = np.concatenate((self.data_features,tmp_features))
                    self.data_truth_label = np.concatenate((self.data_truth_label,tmp_truth_label))
                    self.data_jettype = np.concatenate((self.data_jettype, tmp_jettype))
                    self.data_vartype = np.concatenate((self.data_vartype, tmp_vartype))
            #print("vartypes",self.data_vartype[:100])
            #print("jettypes",self.data_jettype[:10],self.data_jettype.shape)
            if self.dry_run or self.data_vartype.shape[0]>self.Nmaxsample:
                print(f'done loading {self.data_vartype.shape[0]}')
                break
                
     
                



#     def download(self):
#         raise RuntimeError(
#             'Dataset not found. Please download it from {} and move all '
#             '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return self.data_features.shape[0]

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []

    def get(self, idx):
      

        edge_index = torch.empty((2,0), dtype=torch.long)
        if self.istransformer:
            Npfs = 100 
            x_pf = self.data_features[idx,:,:]
            x = torch.from_numpy(self.data_features[idx,:,:])
        else:
            #Removing zero padding for graph data
            Npfs = np.count_nonzero(self.data_features[idx,:,0])
            
            if self.kinematics_only:         
                x_pf = self.data_features[idx,:Npfs,:4]
                x = torch.from_numpy(self.data_features[idx,:Npfs,:4])
            else:
                x_pf = self.data_features[idx,:Npfs,:]
                x = torch.from_numpy(self.data_features[idx,:Npfs,:])
       
   
        # convert to torch
        x_pf = torch.from_numpy(x_pf).float()
     
        #print(x_pf.shape)
        # targets
        y = torch.from_numpy(np.asarray(self.data_truth_label[idx]))
        jet_type = torch.from_numpy(np.asarray(self.data_jettype[idx]))
        vartype = torch.from_numpy(np.asarray(self.data_vartype[idx]))
        #print(y) 
        #x_part = torch.from_numpy(np.asarray(self.data_parton_features[idx]))
#         print(x)
#         print(y)
#         print(vartype)
#         print(jet_type)
        return Data(x=x, edge_index=edge_index, y=y, vartype=vartype, 
                        x_pf=x_pf,jet_type = jet_type)# x_part=x_part)

