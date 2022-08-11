import os.path as osp
import glob

import h5py
import numpy as np
from tqdm import tqdm
import uproot

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class CLV1ROOT(Dataset):
    r'''
        input x0: (PF candidates)

        # truth targets = event class (S vs B)
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, ratio=False):
        super(CLV1ROOT, self).__init__(root, transform)
        
        self.strides = [0]
        self.ratio = ratio
        self.calculate_offsets()

    def calculate_offsets(self):
        for path in self.raw_paths:
            tmptree = uproot.open(path)['events']
            self.strides.append(tmptree.num_entries)
        self.strides = np.cumsum(self.strides)
        print(self.strides)
        
    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return self.strides[-1]

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.root')))
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        #print("HIIIIII")
        file_idx = np.searchsorted(self.strides, idx) - 1
        if file_idx < 0:
            file_idx = 0
        idx_in_file = idx - self.strides[max(0, file_idx)]  # - 1

        if idx_in_file in self.strides and idx_in_file != 0:
            idx_in_file = 0
            file_idx += 1
        
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        edge_index = torch.empty((2,0), dtype=torch.long)

        tmptree = uproot.open(self.raw_paths[file_idx])['events']

        '''
        print("#####")
        
        print("file_idx")
        print(file_idx)
        print("file")
        print(self.raw_paths[file_idx])
        print("idx_in_file")
        print(idx_in_file)
        '''
        
        feat_pt = tmptree['pfcand_pt'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_relpt = tmptree['pfcand_relpt'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_eta = tmptree['pfcand_eta'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_phi = tmptree['pfcand_phi'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_e = tmptree['pfcand_e'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_rele = tmptree['pfcand_rele'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_charge = tmptree['pfcand_charge'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_pdgid = tmptree['pfcand_pdgid'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_d0 = tmptree['pfcand_d0'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_dz = tmptree['pfcand_dz'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_dr = tmptree['pfcand_dr'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        feat_ispho = (feat_pdgid == 22)
        feat_ismu = (np.abs(feat_pdgid) == 11)
        feat_isel = (np.abs(feat_pdgid) == 13)
        feat_isch = (np.abs(feat_pdgid) == 211) | (np.abs(feat_pdgid) == 321) | (np.abs(feat_pdgid) == 2212)  | (np.abs(feat_pdgid) == 3112) | (np.abs(feat_pdgid) == 3222) | (np.abs(feat_pdgid) == 3312) | (np.abs(feat_pdgid) == 3334  )
        feat_isnh = (np.abs(feat_pdgid) == 0)
        
        jet_pt = tmptree['jet_pt'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        jet_eta = tmptree['jet_eta'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        jet_phi = tmptree['jet_phi'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        jet_msd = tmptree['jet_msd'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        jet_n2 = tmptree['jet_n2'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        jet_type = tmptree['jettype'].array(entry_start=idx_in_file,entry_stop=idx_in_file+1)[0]
        
        #'pfcand_relpt','pfcand_eta','pfcand_phi','pfcand_e','pfcand_rele','pfcand_charge','pfcand_pdgid','pfcand_d0','pfcand_dz','pfcand_dr'

        #print(jet_n2)
        #print(jet_type)

        jet_feat = np.array([jet_pt,jet_eta,jet_phi,jet_msd,jet_n2])        
        
        pfcand_feat = []
        
        for pfcand in range(len(feat_pt)):
            tmpfeat = [feat_pt[pfcand],feat_relpt[pfcand],feat_eta[pfcand],\
                           feat_phi[pfcand],feat_e[pfcand],feat_rele[pfcand],\
                           feat_charge[pfcand],feat_d0[pfcand],\
                           feat_dz[pfcand],feat_dr[pfcand],feat_ispho[pfcand],feat_ismu[pfcand],feat_isel[pfcand],feat_isch[pfcand],feat_isnh[pfcand]]

            pfcand_feat.append(tmpfeat)

        pfcand_feat = np.array(pfcand_feat)
        
        #print(np.array(feat_pt))
        #print(np.array(feat_pt).shape)
        #print(pfcand_feat)
        
        Npfs = np.count_nonzero(pfcand_feat[:,0])
        x_pf = pfcand_feat[:Npfs,:]
        
        # convert to torch
        x = torch.from_numpy(pfcand_feat[:Npfs,:])
        x_pf = torch.from_numpy(x_pf).float()

        x_jet = torch.from_numpy(jet_feat)

        y = torch.from_numpy(np.array(jet_type))

        return Data(x=x, edge_index=edge_index, y=y,
                        x_pf=x_pf, x_jet=x_jet)
        

