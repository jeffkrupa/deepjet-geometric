import sklearn
import numpy as np
import pandas as pd

from deepjet_geometric.datasets import TICLV1
from torch_geometric.data import DataLoader

import os
import argparse

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
parser.add_argument('--model', action='store', type=str, help='Path to folder with best-epoch.pt file.')
args = parser.parse_args()

BATCHSIZE = 128

data_train = TICLV1(args.ipath,ratio=False)

val_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_ts', 'x_lc'])

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear

model_dir = args.opath


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 16
        
        self.ts_encode = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.lc_encode = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=16
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 4),
            nn.ELU(),
            nn.Linear(4, 1)
        )
        
    def forward(self,
                x_ts, x_lc,
                batch_ts, batch_lc):
        #print(x_ts)
        x_ts_enc = self.ts_encode(x_ts)
        x_lc_enc = self.lc_encode(x_lc)
        

        # create a representation of LCs to LCs
        feats1 = self.conv(x=(x_lc_enc, x_lc_enc), batch=(batch_lc, batch_lc))
        # similarly a representation LCs to Trackster
        feats2 = self.conv(x=(x_lc_enc, x_ts_enc), batch=(batch_lc, batch_ts))
        # and now from the LCs-to-LCs to the Trackster-to-LCs
        feats3 = self.conv(x=(feats1, feats2), batch=(batch_lc, batch_ts))

        #batch = batch_ts
        out, batch = avg_pool_x(batch_ts, feats3, batch_ts)
        out = self.output(out)
        
        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

ticl = Net().to(device)
ticl.load_state_dict(torch.load(model_dir+"best-epoch.pt")['model'])
optimizer = torch.optim.Adam(ticl.parameters(), lr=0.001)
optimizer.load_state_dict(torch.load(model_dir+"best-epoch.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
scheduler.load_state_dict(torch.load(model_dir+"best-epoch.pt")['lr'])


@torch.no_grad()
def test():
    ticl.eval()
    total_loss = 0
    counter = 0
    
    rawdata = [] # raw energy, truth energy, regressed regressed, truth pid, regressed pid
    
    for data in val_loader:
        counter += 1
        print(str(counter*BATCHSIZE)+' / '+str(len(val_loader.dataset)),end='\r')
        data = data.to(device)
        with torch.no_grad():
            out = ticl(data.x_ts,
                       data.x_lc,
                       data.x_ts_batch,
                       data.x_lc_batch)
            
            
            
            for i in range(len(out[0][:,0])): 
                rawdata.append([data.x_ts.float()[:,0][i].item(),data.y[data.y>-1.].float()[::2][i].item(),out[0][:,0][i].item()])
            #print(out[0][:,0]/data.y[data.y>-1.].float()[::2])
            #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].float())
            #total_loss += loss.item()
    df = pd.DataFrame(rawdata, columns=['rawe', 'truthe', 'rege']) 
    df.to_csv(model_dir+"/"+"rawdata.csv")

    return total_loss / len(val_loader.dataset)


for epoch in range(1):
    loss = test()



