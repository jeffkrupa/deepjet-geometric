import sklearn
import numpy as np
from random import randrange
import subprocess
import tqdm
import pandas as pd

from deepjet_geometric.datasets import CLV1
from torch_geometric.data import DataLoader

import os
import argparse

BATCHSIZE = 200

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--vpath', action='store', type=str, help='Path to validation files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
args = parser.parse_args()

print(args.ipath)

data_train = CLV1(args.ipath,ratio=True)
data_test = CLV1(args.vpath,ratio=True)

train_loader = DataLoader(data_train, batch_size=BATCHSIZE,shuffle=False,
                          follow_batch=['x_pf'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=False,
                         follow_batch=['x_pf'])

#exit(1)

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
#from torch_geometric.graphgym.models.pooling import global_add_pool
from torch.nn import Sequential, Linear
from torch_geometric.nn import DataParallel
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter

model_dir = args.opath

def global_add_pool(x, batch, size=None):
    """
    Globally pool node embeddings into graph embeddings, via elementwise sum.
    Pooling function takes in node embedding [num_nodes x emb_dim] and
    batch (indices) and outputs graph embedding [num_graphs x emb_dim].

    Args:
        x (torch.tensor): Input node embeddings
        batch (torch.tensor): Batch tensor that indicates which node
        belongs to which graph
        size (optional): Total number of graphs. Can be auto-inferred.

    Returns: Pooled graph embeddings

    """
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICRegLoss(torch.nn.Module):

    def __init__(self, lambda_param=1,mu_param=1,nu_param=20):
        super(VICRegLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        #self.device = torch.device('cpu')

    def forward(self, x, y):
        
        self.device = (torch.device('cuda')if x.is_cuda else torch.device('cpu'))
        x = F.normalize( x, dim=1 )
        y = F.normalize( y, dim=1 )
        x_scale = x
        y_scale = y
        repr_loss = F.mse_loss(x_scale, y_scale)
        
        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x_scale = x_scale - x_scale.mean(dim=0)
        y_scale = y_scale - y_scale.mean(dim=0)
        N = x_scale.size(0)
        D = x_scale.size(1)
        
        std_x = torch.sqrt(x_scale.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y_scale.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        x_scale = (x_scale - x_scale.mean(dim=0))/x_scale.std(dim=0) ##!!!!! More Robust
        y_scale = (y_scale - y_scale.mean(dim=0))/y_scale.std(dim=0) ##!!!!! More Robust
        cov_x = (x_scale.T @ x_scale) / (N - 1)
        cov_y = (y_scale.T @ y_scale) / (N - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D)

        #loss = (self.lambda_param * repr_loss + self.mu_param * std_loss+ self.nu_param * cov_loss)
        #print(repr_loss,cov_loss,std_loss)
        return repr_loss,cov_loss,std_loss








class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 128
        
        self.pf_encode = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ELU(),
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
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 8)#,
#            nn.ELU(),
#            nn.Linear(16, 8)
        )
        
    def forward(self,
                x_pf,
                batch_pf):
        #print(x_ts)
        #x_pf = BatchNorm(x_pf)
        x_pf_enc = self.pf_encode(x_pf)
        
        # create a representation of LCs to LCs
        feats1 = self.conv(x=(x_pf_enc, x_pf_enc), batch=(batch_pf, batch_pf))
        feats2 = self.conv(x=(feats1, feats1), batch=(batch_pf, batch_pf))
        # similarly a representation LCs to Trackster
        feats3 = self.conv(x=(feats2, feats2), batch=(batch_pf, batch_pf))


        batch = batch_pf
        #out, batch = avg_pool_x(batch_pf, feats3, batch_pf)
        out  = global_add_pool(feats3, batch_pf)
        out = self.output(out)

        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

cl = Net().to(device)
#cl = DataParallel(cl)

#puma.load_state_dict(torch.load(model_dir+"epoch-32.pt")['model'])
optimizer = torch.optim.Adam(cl.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(model_dir+"epoch-32.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#scheduler.load_state_dict(torch.load(model_dir+"epoch-32.pt")['lr'])

vrloss = VICRegLoss()

def train():
    cl.train()
    counter = 0

    total_loss = 0
    for data in tqdm.tqdm(train_loader):
        counter += 1

        #print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        optimizer.zero_grad()
        out = cl(data.x_pf,
                    data.x_pf_batch)

        #out[0][:, 1] = torch.sigmoid(out[0][:, 1])
        '''
        print(out[0])
        print("A")
        print(out[0])
        print("B")
        print(out[0][0::2])
        print("C")
        print(out[0][1::2])
        print("A parton")
        print(data.x_part)
        print("B parton")
        print(data.x_part[0::2])
        print("C parton")
        print(data.x_part[1::2])
        #print(data.y)
        '''
        #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].float())
        #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].reshape(-1,2)[:,:1].reshape(-1))
        #loss = contrastive_loss(out[0][0::2],out[0][1::2],0.1)
        loss_clr,loss_corr,loss_var = vrloss(out[0][0::2],out[0][1::2]) 
        loss = loss_clr + loss_corr + loss_var
        #print(data.y)
        #print(loss.item())
        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        #if counter > 1:
        #    break
        
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    cl.eval()
    total_loss = 0
    counter = 0
    for data in tqdm.tqdm(test_loader):
        counter += 1
        #print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        with torch.no_grad():
            out = cl(data.x_pf,
                       data.x_pf_batch)

            #out[0][:, 1] = torch.sigmoid(out[0][:, 1])
            #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].float())
            #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].reshape(-1,2)[:,:1].reshape(-1))
            #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1),data.y[data.y>-1.].float())
            #loss = contrastive_loss(out[0][0::2],out[0][1::2],0.1)
            loss_clr,loss_corr,loss_var = vrloss(out[0][0::2],out[0][1::2])
            loss = loss_clr + loss_corr + loss_var

            total_loss += loss.item()
    return total_loss / len(test_loader.dataset)

best_val_loss = 1e9

all_train_loss = []
all_val_loss = []

loss_dict = {'train_loss': [], 'val_loss': []}

for epoch in range(1, 100):
    print(f'Training Epoch {epoch} on {len(train_loader.dataset)} jets')
    loss = train()
    scheduler.step()

    #exit(1)
    
    print(f'Validating Epoch {epoch} on {len(test_loader.dataset)} jets')
    loss_val = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    all_train_loss.append(loss)
    all_val_loss.append(loss_val)
    loss_dict['train_loss'].append(loss)
    loss_dict['val_loss'].append(loss_val)
    df = pd.DataFrame.from_dict(loss_dict)
    

    if not os.path.exists(model_dir):
        subprocess.call("mkdir -p %s"%model_dir,shell=True)

    df.to_csv("%s/"%model_dir+"/loss.csv")

    state_dicts = {'model':cl.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}

    torch.save(state_dicts, os.path.join(model_dir, f'epoch-{epoch}.pt'))
    
    if loss_val < best_val_loss:
        best_val_loss = loss_val

        torch.save(state_dicts, os.path.join(model_dir, 'best-epoch.pt'.format(epoch)))


print(all_train_loss)
print(all_val_loss)

