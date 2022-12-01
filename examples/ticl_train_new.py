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

BATCHSIZE = 64

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--vpath', action='store', type=str, help='Path to validation files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
args = parser.parse_args()

print(args.ipath)

data_train = TICLV2(args.ipath,hierarchy=2,ratio=True)
data_test = TICLV2(args.vpath,hierarchy=2,ratio=True)

train_loader = DataLoader(data_train, batch_size=BATCHSIZE,shuffle=False,
                          follow_batch=['x_ts', 'x_lc', 'x_ass'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=False,
                         follow_batch=['x_ts', 'x_lc', 'x_ass'])

#exit(1)

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
        #print(x_ts)
        x_ts_enc = self.ts_encode(x_ts)
        x_lc_enc = self.lc_encode(x_lc)
        
        # create a representation of LCs to LCs
        feats1 = self.conv(x=(x_lc_enc, x_lc_enc), batch=(batch_lc, batch_lc))
        feats11 = self.conv(x=(feats1, feats1), batch=(batch_lc, batch_lc))
        # create a representation of Tracksters to Tracksters
        feats2 = self.conv(x=(x_ts_enc, x_ts_enc), batch=(batch_ts, batch_ts))
        feats22 = self.conv(x=(feats2, feats2), batch=(batch_ts, batch_ts))
        # similarly a representation LCs to Tracksters
        feats3 = self.conv(x=(feats22, feats11), batch=(batch_ts, batch_lc))


        #batch = batch_ts
        out, batch = avg_pool_x(batch_lc, feats3, batch_lc)
        out = self.output(out)

        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

ticl = Net().to(device)
#ticl.load_state_dict(torch.load(model_dir+"best-epoch.pt")['model'])
optimizer = torch.optim.Adam(ticl.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(model_dir+"best-epoch.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#scheduler.load_state_dict(torch.load(model_dir+"best-epoch.pt")['lr'])


def train():
    ticl.train()
    counter = 0

    total_loss = 0
    for data in tqdm.tqdm(train_loader):
        counter += 1

        data = data.to(device)
        optimizer.zero_grad()
        out = ticl(data.x_ts,
                    data.x_lc,
                    data.x_ass,
                    data.x_ts_batch,
                    data.x_lc_batch,
                    data.x_ass_batch)

        energy_loss = nn.MSELoss()(torch.squeeze(out[0][:,0]).view(-1),data.y[::2].float())
        class_loss = nn.CrossEntropyLoss()(out[0][:,1:],data.y[1::2].long())
        loss = energy_loss + class_loss

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    ticl.eval()
    total_loss = 0
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

    return total_loss / len(test_loader.dataset)

best_val_loss = 1e9
loss_dict = {'train_loss': [], 'val_loss': []}

for epoch in range(1, 100):
    loss = train()
    scheduler.step()

    loss_val = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    if not os.path.exists(model_dir):
        subprocess.call("mkdir -p %s"%model_dir,shell=True)

    loss_dict['train_loss'].append(loss)
    loss_dict['val_loss'].append(loss_val)
    df = pd.DataFrame.from_dict(loss_dict)

    df.to_csv("%s/"%model_dir+"/loss.csv")

    state_dicts = {'model':ticl.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()}

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-%i.pt'%epoch))
    
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        
        torch.save(state_dicts, os.path.join(model_dir, 'best-epoch.pt'.format(epoch)))

