import os.path as osp
import glob

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

def files_exist(files):
    return len(files) != 0 and all(osp.exists(f) for f in files)


class TICLV2(Dataset):
    r'''
        input x0: (Tracksters)
               
        input x1: (LayerClusters beloging to Tracksters)

        # truth targets = Energy and PID
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, ratio=False, hierarchy=1):
        super(TICLV2, self).__init__(root, transform)
        
        self.strides = [0]
        self.ratio = ratio
        self.calculate_offsets()
        self.hierarchy = hierarchy
        #print("YYYYYYYY")
        #print(self.raw_paths)

    def calculate_offsets(self):
        for path in self.raw_paths:
            #print("WTFFFFF")
            with h5py.File(path, 'r') as f:
                self.strides.append(f['tss'][()].shape[0])
        self.strides = np.cumsum(self.strides)

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return self.strides[-1]

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        #print(raw_files)
        #print(files_exist(raw_files))
        #print(self.__class__.__dict__.keys())
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx) - 1
        idx_in_file = idx - self.strides[max(0, file_idx)] - 1


        #print(file_idx)
        #print(idx_in_file)
        #print("___")   
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        edge_index = torch.empty((2,0), dtype=torch.long)
        with h5py.File(self.raw_paths[file_idx],'r') as f:
            
            Nlc = np.count_nonzero(f['lcs'][idx_in_file,:,0])
            Nts = np.count_nonzero(f['tssCLUE3D'][idx_in_file,:,0])
            
            x_lc = f['lcs'][idx_in_file,:Nlc,:]
            x_ts = f['tssCLUE3D'][idx_in_file,:Nts,:]
            x_ass = f['tss'][idx_in_file,:]

            cps = f['cp'][idx_in_file,:]
            #print(cps)
            #print(x_ts)
            #cps[0] = np.sum(x_ts[:,0].flatten())/cps[0]
            if self.ratio:
                cps[0] = cps[0]/x_ass[0]
                #print(x_ass[0])
            else:
                cps[0] = np.log(cps[0])
            cps[1] = np.where(cps[1] == 11,0,cps[1])
            cps[1] = np.where(cps[1] == 22,1,cps[1]) 
            cps[1] = np.where(cps[1] == 130,2,cps[1])
            cps[1] = np.where(cps[1] == 211,3,cps[1])

            #print(cps[1])
            is_electron = np.reshape((cps[1] == 0), (1,1))
            is_photon = np.reshape((cps[1] == 1), (1,1))
            is_kaon = np.reshape((cps[1] == 2), (1,1))
            is_pion = np.reshape((cps[1] == 3), (1,1))

            classtype = np.hstack((is_electron,is_photon,is_kaon,is_pion))

            rawe = np.sum(x_ts[:,0].flatten())
            rawe = torch.from_numpy(np.array([rawe])).float()

            # convert to torch
            x = torch.from_numpy(f['lcs'][idx_in_file,:][None]).float()
            x_ts = torch.from_numpy(x_ts).float()
            x_lc = torch.from_numpy(x_lc).float()
            x_ass = torch.from_numpy(x_ass).float()
            # targets

            #f['cp'][idx_in_file,1] = np.where(np.abs(f['cp'][idx_in_file,1]) == 211,1,0)

            y = torch.from_numpy(cps).float()
            classtype = torch.from_numpy(classtype).long()

            if self.hierarchy == 1:
                return Data(x=x, edge_index=edge_index, y=y,
                            x_lc=x_lc, x_ts=x_ts, rawe=rawe, classtype=classtype)

            if self.hierarchy == 2:
                return Data(x=x, edge_index=edge_index, y=y,
                        x_lc=x_lc, x_ts=x_ts, rawe=rawe, x_ass=x_ass[None,:])




