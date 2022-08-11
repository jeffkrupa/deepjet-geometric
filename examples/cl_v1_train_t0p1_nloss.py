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

BATCHSIZE = 1024

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


def contrastive_loss( x_i, x_j, temperature=0.1 ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 )
    z_j = F.normalize( x_j, dim=1 )
    #print("___")
    #print(x_i)
    #print(x_j)
    #z_i = x_i
    #z_j = x_j
    z   = torch.cat( [z_i, z_j], dim=0 )
    #print(z)
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    #print(similarity_matrix)
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 )
    nominator = torch.exp( positives / temperature )
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    loss_partial = -torch.log( nominator / torch.sum( denominator, dim=1 ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss



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
        loss = contrastive_loss(out[0][0::2],out[0][1::2],0.1)

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
            loss = contrastive_loss(out[0][0::2],out[0][1::2],0.1)


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

    if loss_val < best_val_loss:
        best_val_loss = loss_val
        
        state_dicts = {'model':cl.state_dict(),
                       'opt':optimizer.state_dict(),
                       'lr':scheduler.state_dict()} 

        torch.save(state_dicts, os.path.join(model_dir, 'best-epoch.pt'.format(epoch)))


print(all_train_loss)
print(all_val_loss)

