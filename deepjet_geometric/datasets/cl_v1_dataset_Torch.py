import os,sys
import os.path as osp
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class CLV1_Torch(Dataset):
    """
    PyTorch Dataset for loading data on-the-fly.
    """

    def __init__(self, root, Nevents=None, transform=None, ratio=False, seedOnly=False, opath=None):
        self.root = root
        self.Nevents = Nevents
        self.transform = transform
        self.ratio = ratio
        self.seedOnly = seedOnly
        self.strides = [0]
        self.calculate_offsets()
        '''
        stdoutOrigin=sys.stdout

        sys.stdout = open("./"+opath+"_log.txt", "w")
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
        print("# of Wqq jets={}, {:.2f}%".format(np.sum(self.data_jettype==11), 100*np.sum(self.data_jettype==11)/len(self.data_jettype)))

        print("# of nominal jets={}, {:.2f}%".format(np.sum(self.data_vartype==-1), 100*np.sum(self.data_vartype==-1)/len(self.data_vartype)))
        print("# of seed jets={}, {:.2f}%".format(np.sum(self.data_vartype==0), 100*np.sum(self.data_vartype==0)/len(self.data_vartype)))
        print("# of fsrLo jets={}, {:.2f}%".format(np.sum(self.data_vartype==1), 100*np.sum(self.data_vartype==1)/len(self.data_vartype)))
        print("# of fsrHi jets={}, {:.2f}%".format(np.sum(self.data_vartype==2), 100*np.sum(self.data_vartype==2)/len(self.data_vartype)))
        print("# of herwig jets={}, {:.2f}%".format(np.sum(self.data_vartype==3), 100*np.sum(self.data_vartype==3)/len(self.data_vartype)))
        sys.stdout.close()
        sys.stdout=stdoutOrigin
        '''

    def calculate_offsets(self):
        self.raw_paths = sorted(glob.glob(osp.join(self.root, '*.h5')))
        total_events = 0
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                num_events = f['features'].shape[0]
                self.strides.append(num_events)
                total_events += num_events
                if self.Nevents is not None and total_events >= self.Nevents:
                    break
        self.strides = np.cumsum(self.strides)

    def __len__(self):
        if self.Nevents is not None:
            return min(self.Nevents, self.strides[-1])
        return self.strides[-1]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError(f'Index {idx} is out of bounds for dataset of length {self.__len__()}')

        file_idx = np.searchsorted(self.strides, idx) - 1
        idx_in_file = idx - self.strides[max(0, file_idx)]
        print("idx",idx)
        print("file_idx",file_idx)
        print("idx_in_file",idx_in_file)
        print("self.raw_paths[file_idx]",self.raw_paths[file_idx])
        with h5py.File(self.raw_paths[file_idx], 'r') as f:
            #Npfs = np.count_nonzero(f['features'][idx_in_file, :, 0])

            print("f['features'][idx_in_file, :, :]).float().shape",torch.from_numpy(f['features'][idx_in_file, :, :]).float().shape)
            x_pf = torch.from_numpy(f['features'][idx_in_file, :, :]).float()
            #y = torch.tensor(f['truth_label'][idx_in_file], dtype=torch.float32)
            #x_vartype = torch.tensor(f['vartype'][idx_in_file], dtype=torch.float32)
            #x_jettype = torch.tensor(f['jettype'][idx_in_file], dtype=torch.float32)
            #x_part = torch.from_numpy(f['parton_features'][idx_in_file]).float()
            #x_jet = torch.from_numpy(f['jet_features'][idx_in_file]).float()
            #print(x_pf,y,x_vartype)
            # Apply any transformations

            return {'x_pf' : x_pf}

# Usage example
# dataset = CLV1(root='/path/to/dataset', Nevents=10000)
# loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

