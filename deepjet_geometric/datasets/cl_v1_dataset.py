import os.path as osp
import glob

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class CLV1(Dataset):
    r'''
        input x0: (PF candidates)

        # truth targets = event class (S vs B)
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, ratio=False):
        super(CLV1, self).__init__(root, transform)
        
        self.strides = [0]
        self.ratio = ratio
        self.calculate_offsets()

    def calculate_offsets(self):
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(f['features'].shape[0])
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
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx) - 1
        if file_idx < 0:
            file_idx = 0
        idx_in_file = idx - self.strides[max(0, file_idx)]  # - 1

        if idx in self.strides and idx_in_file != 0:
            idx_in_file = 0
            file_idx += 1
        
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        edge_index = torch.empty((2,0), dtype=torch.long)
        with h5py.File(self.raw_paths[file_idx],'r') as f:

            '''
            print("===========================")
            print("===========================")
            print("=== strides")
            print(self.strides)
            print("=== idx")
            print(idx)
            print("=== file")
            print(self.raw_paths[file_idx])
            print("=== file idx")
            print(file_idx)
            print("=== idx in file")
            print(idx_in_file)
            print("Shape")
            print(f['features'][()].shape)
            #print(np.count_nonzero(f['features'][110148,:,0]))
            '''
            
            Npfs = np.count_nonzero(f['features'][idx_in_file,:,0])
            
            
            x_pf = f['features'][idx_in_file,:Npfs,:]

            # convert to torch
            x = torch.from_numpy(f['features'][idx_in_file,:Npfs,:])
            x_pf = torch.from_numpy(x_pf).float()

            # targets
            y = torch.from_numpy(np.asarray(f['truth_label'][idx_in_file]))

            x_part = torch.from_numpy(np.asarray(f['parton_features'][idx_in_file]))

            #print("=== x_part")
            #print(x_part)
            
            return Data(x=x, edge_index=edge_index, y=y,
                        x_pf=x_pf, x_part=x_part)

