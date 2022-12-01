import sklearn
import numpy as np
from random import randrange
import subprocess

from deepjet_geometric.datasets import TICLV2
from torch_geometric.data import DataLoader

import os
import argparse
import tqdm
import pandas as pd

BATCHSIZE = 1

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save inferred files.')
parser.add_argument('--mpath', action='store', type=str, help='Path to folder with best-epoch.pt file.')
args = parser.parse_args()

print(args.ipath)
data_test = TICLV2(args.ipath,ratio=True,hierarchy=2)
test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=False,
                         follow_batch=['x_ts', 'x_lc', 'x_ass'])

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn import DataParallel

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 128
        
        self.ts_encode = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.lc_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 5)
        )
        
    def forward(self,
                x_ts, x_lc, x_ass,
                batch_ts, batch_lc, batch_ass):

        x_ts_enc = self.ts_encode(x_ts)
        x_lc_enc = self.lc_encode(x_lc)
        

        feats1 = self.conv(x=(x_lc_enc, x_lc_enc), batch=(batch_lc, batch_lc))
        feats11 = self.conv(x=(feats1, feats1), batch=(batch_lc, batch_lc))
        feats2 = self.conv(x=(x_ts_enc, x_ts_enc), batch=(batch_ts, batch_ts))
        feats22 = self.conv(x=(feats2, feats2), batch=(batch_ts, batch_ts))
        feats3 = self.conv(x=(feats22, feats11), batch=(batch_ts, batch_lc))

        out, batch = avg_pool_x(batch_lc, feats3, batch_lc)
        out = self.output(out)

        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

ticl = Net().to(device)
ticl.load_state_dict(torch.load(args.mpath+"/best-epoch.pt")['model'])

@torch.no_grad()
def test():
    ticl.eval()
    total_loss = 0

    rawdata = [] # raw energy, truth energy, regressed regressed, truth pid, regressed pid

    counter = 0
    for data in tqdm.tqdm(test_loader):
        counter += 1
        data = data.to(device)
        with torch.no_grad():
            out = ticl(data.x_ts,
                       data.x_lc,
                       data.x_ass,
                       data.x_ts_batch,
                       data.x_lc_batch,
                       data.x_ass_batch)
   
            energy_loss = nn.MSELoss()(torch.squeeze(out[0][:,0]).view(-1),data.y[::2].float())
            class_loss = nn.CrossEntropyLoss()(out[0][:,1:],data.y[1::2].long())

            loss = energy_loss + class_loss

            total_loss += loss.item()

            for i in range(len(out[0][:,0])): 
                rawdata.append([data.x_ass[:,0].cpu().numpy()[0],torch.sum(data.x_ts[:,0]).cpu().numpy(),data.x_ass[:,0].cpu().numpy()[0]*data.y[::2].cpu().numpy()[0],data.x_ass[:,0].cpu().numpy()[0]*torch.squeeze(out[0][:,0]).view(-1).cpu().numpy()[0],
                    np.argmax(torch.squeeze(out[0][:,1:]).view(-1).cpu().numpy()),data.y[1::2].cpu().numpy()[0]])

    return total_loss / len(test_loader.dataset), rawdata

for epoch in range(0, 1):

    print("Running")
    loss_val, rawdata = test()

    print('Loss: {:.8f}'.format(loss_val))

    if not os.path.exists(args.opath):
        subprocess.call("mkdir -p %s"%args.opath,shell=True)
    
    df = pd.DataFrame(rawdata, columns=['asse', 'rawe', 'truthe', 'rege', 'truthpid', 'redpid'])
    df.to_csv(os.path.join(args.opath,"rawdata.csv"))






