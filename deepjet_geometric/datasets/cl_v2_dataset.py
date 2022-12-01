import os.path as osp
import glob

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class CLV2(Dataset):
    r'''
        input x0: (PF candidates)

        # truth labels = jet class
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None):
        super(CLV2, self).__init__(root, transform)
        
        self.fill_data()

    def fill_data(self):
        for fi,path in enumerate(self.raw_paths):
            with h5py.File(path, 'r') as f:
                 tmp_features = f['features'][()]
                 tmp_truth_label = f['truth_label'][()]
                 tmp_parton_features = f['parton_features'][()]
                 if fi == 0:
                     self.data_features = tmp_features
                     self.data_truth_label = tmp_truth_label
                     self.data_parton_features = tmp_parton_features
                 else:
                     self.data_features = np.concatenate((self.data_features,tmp_features))
                     self.data_truth_label = np.concatenate((self.data_truth_label,tmp_truth_label))
                     self.data_parton_features = np.concatenate((self.data_parton_features,tmp_parton_features))

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

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

        Npfs = np.count_nonzero(self.data_features[idx,:,0])
                
        x_pf = self.data_features[idx,:Npfs,:]

        # convert to torch
        x = torch.from_numpy(self.data_features[idx,:Npfs,:])
        x_pf = torch.from_numpy(x_pf).float()

        # targets
        y = torch.from_numpy(np.asarray(self.data_truth_label[idx]))

        x_part = torch.from_numpy(np.asarray(self.data_parton_features[idx]))

        return Data(x=x, edge_index=edge_index, y=y,
                        x_pf=x_pf, x_part=x_part)

