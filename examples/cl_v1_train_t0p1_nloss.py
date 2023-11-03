import sklearn
import numpy as np
from random import randrange
import subprocess
import tqdm
import pandas as pd

from deepjet_geometric.datasets import CLV2
from torch_geometric.data import DataLoader

import os
import argparse

BATCHSIZE = 200

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--vpath', action='store', type=str, help='Path to validation files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
parser.add_argument('--temperature', action='store', type=str, help='SIMCLR Temperature.')
parser.add_argument('--hidden_dim', action='store', type=int, help='Hidden dimension.')
parser.add_argument('--nepochs', action='store', type=str, help='Number of epochs to train for.')
parser.add_argument('--n_out_nodes', action='store', type=int, help='Number of output (encoded) nodes.')
parser.add_argument('--qcd_only', action='store_true',default=False, help='Run on QCD only.')
parser.add_argument('--seed_only', action='store_true',default=False, help='Run on seed only.')
parser.add_argument('--dry_run', action='store_true',default=False, help='Only run on  one file.')
parser.add_argument('--herwig_only', action='store_true',default=False, help='Run on herwig only.')
parser.add_argument('--mpath', action='store', type=str, help='If specified will load model')
parser.add_argument('--abseta', action='store_true',default=False, help='Run on abseta.')
parser.add_argument('--which_augmentations', action='store',type=int,nargs='+',default=None, help='Run on these augmentations (0=seed, 1=fsrUp, 2=fsrDown, 3=herwig7)')
parser.add_argument('--kinematics_only', action='store_true',default=False, help='Train on kinematics only.')

parser.add_argument('--continue_training',action='store_true',default=False)
args = parser.parse_args()
temperature = float(args.temperature)
nepochs = int(args.nepochs)
print(args.ipath)
print("qcd only? ", args.qcd_only)
print("seed only? ", args.seed_only)
print("train with kinematics only? ", args.kinematics_only)
print("train with abseta? ", args.abseta)
print("which augmentations? " , args.which_augmentations)
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)
    #subprocess.call("mkdir -p %s"%model_dir,shell=True)
data_train = CLV2(args.ipath,args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run)
data_test = CLV2(args.vpath,args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/test", args.dry_run)

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
    #print("temperature",temperature)
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
    def __init__(self,n_in_nodes,n_out_nodes,hidden_dim):
        super(Net, self).__init__()
        print("n_in_nodes",n_in_nodes)
        print("n_out_nodes",n_out_nodes)
        self.hidden_dim = hidden_dim
        self.n_out_nodes = n_out_nodes 
        self.pf_encode = nn.Sequential(
            nn.Linear(n_in_nodes, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
     
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )
        self.conv3 = DynamicEdgeConv(
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
            nn.Linear(32, self.n_out_nodes)#,
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
        feats1 = self.conv1(x=(x_pf_enc, x_pf_enc), batch=(batch_pf, batch_pf))
        feats2 = self.conv2(x=(feats1, feats1), batch=(batch_pf, batch_pf))
        # similarly a representation LCs to Trackster
        feats3 = self.conv3(x=(feats2, feats2), batch=(batch_pf, batch_pf))


        batch = batch_pf
        #out, batch = avg_pool_x(batch_pf, feats3, batch_pf)
        out  = global_add_pool(feats3, batch_pf)
      
        out = self.output(out)

        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
n_out_nodes = int(args.n_out_nodes)
if args.kinematics_only:
    n_in_nodes=4
else:
    n_in_nodes=15
cl = Net(n_in_nodes,n_out_nodes,args.hidden_dim).to(device)
#cl = DataParallel(cl)

def load_matching_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)['model']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict,strict=False)

#Loading Model
if args.continue_training:
    cl.load_state_dict(torch.load(args.mpath)['model'])
    start_epoch = args.mpath.split("/")[-1].split("-")[-1].split(".")[0]
    start_epoch = int(start_epoch) + 1
    print(f"Continuing training from epoch {start_epoch}...")
#elif args.mpath:
#    load_matching_state_dict(cl,args.mpath)
#    print('loaded model')
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
        #print(data.x_pf.shape, data.x_pf_batch.shape)
        out = cl(data.x_pf,
                    data.x_pf_batch)
        #print("out[0].shape",out[0].shape)
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
        loss = contrastive_loss(out[0][0::2],out[0][1::2],temperature)

        #print(data.y)
        #print(loss.item())
        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        #if counter > 100:
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
            loss = contrastive_loss(out[0][0::2],out[0][1::2],temperature)

       
            total_loss += loss.item()

        #if counter > 100:
        #    break
    return total_loss / len(test_loader.dataset)

best_val_loss = 1e9

all_train_loss = []
all_val_loss = []

loss_dict = {'train_loss': [], 'val_loss': []}

if args.continue_training:
    loss_dict = pd.read_csv(os.path.join(args.mpath.split("/")[0],'loss.csv')).to_dict()
    print(loss_dict['train_loss'].keys())
else:
    start_epoch = 1
    loss_dict = {'train_loss': [], 'val_loss': []}
for epoch in range(start_epoch, nepochs):
#for epoch in range(1, nepochs):
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
    try:
        loss_dict['train_loss'].append(loss)
        loss_dict['val_loss'].append(loss_val)
    except:
        loss_dict['train_loss'][epoch-1] = loss
        loss_dict['val_loss'][epoch-1] = loss_val
    df = pd.DataFrame.from_dict(loss_dict)
    


    df.to_csv("%s/"%model_dir+"/loss.csv")
    
    state_dicts = {'model':cl.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}

    torch.save(state_dicts, os.path.join(model_dir, f'epoch-{epoch}.pt'))

    if loss_val < best_val_loss:
        best_val_loss = loss_val

        torch.save(state_dicts, os.path.join(model_dir, 'best-epoch.pt'.format(epoch)))


print(all_train_loss)
print(all_val_loss)


