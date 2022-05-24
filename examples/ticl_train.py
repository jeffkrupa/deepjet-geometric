import sklearn
import numpy as np
from random import randrange
import subprocess

from deepjet_geometric.datasets import TICLV1
from torch_geometric.data import DataLoader

import os
import argparse

BATCHSIZE = 128

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--vpath', action='store', type=str, help='Path to validation files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
args = parser.parse_args()

print(args.ipath)

data_train = TICLV1(args.ipath,ratio=False)
data_test = TICLV1(args.vpath,ratio=False)

train_loader = DataLoader(data_train, batch_size=BATCHSIZE,shuffle=True,
                          follow_batch=['x_ts', 'x_lc'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=True,
                         follow_batch=['x_ts', 'x_lc'])

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn import DataParallel

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
#puma.load_state_dict(torch.load(model_dir+"epoch-32.pt")['model'])
optimizer = torch.optim.Adam(ticl.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(model_dir+"epoch-32.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#scheduler.load_state_dict(torch.load(model_dir+"epoch-32.pt")['lr'])


def train():
    ticl.train()
    counter = 0

    total_loss = 0
    for data in train_loader:
        counter += 1

        print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        optimizer.zero_grad()
        out = ticl(data.x_ts,
                    data.x_lc,
                    data.x_ts_batch,
                    data.x_lc_batch)
 
        loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].reshape(-1,2)[:,:1].reshape(-1))
        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    ticl.eval()
    total_loss = 0
    counter = 0
    for data in test_loader:
        counter += 1
        print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        with torch.no_grad():
            out = ticl(data.x_ts,
                       data.x_lc,
                       data.x_ts_batch,
                       data.x_lc_batch)
            loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].reshape(-1,2)[:,:1].reshape(-1))
            #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].float())
            total_loss += loss.item()
    return total_loss / len(test_loader.dataset)

best_val_loss = 1e9

for epoch in range(1, 50):
    loss = train()
    scheduler.step()

    loss_val = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    if not os.path.exists(model_dir):
        subprocess.call("mkdir -p %s"%model_dir,shell=True)

    if loss_val < best_val_loss:
        best_val_loss = loss_val
        
        state_dicts = {'model':ticl.state_dict(),
                       'opt':optimizer.state_dict(),
                       'lr':scheduler.state_dict()} 

        torch.save(state_dicts, os.path.join(model_dir, 'best-epoch.pt'.format(epoch)))

