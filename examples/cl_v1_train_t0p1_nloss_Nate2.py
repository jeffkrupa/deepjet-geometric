import sklearn
import numpy as np
from random import randrange
import subprocess
import tqdm
import pandas as pd

from deepjet_geometric.datasets import CLV2_Nate2
from torch_geometric.data import Data

from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from PIL import Image
import os
import time
import sys
import os
import argparse


parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--vpath', action='store', type=str, help='Path to validation files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
parser.add_argument('--temperature', action='store', type=str, help='SIMCLR Temperature.')
parser.add_argument('--lr', action='store', default = 0.001, type=float, help='learning rate')
parser.add_argument('--hidden_dim', action='store', type=int, help='Hidden dimension.')
parser.add_argument('--nepochs', action='store', type=str, help='Number of epochs to train for.')
parser.add_argument('--n_out_nodes', action='store', type=int, help='Number of output (encoded) nodes.')
parser.add_argument('--batchsize', action='store', type=int,default = 200,  help='batch size')
parser.add_argument('--qcd_only', action='store_true',default=False, help='Run on QCD only.')
parser.add_argument('--seed_only', action='store_true',default=False, help='Run on seed only.')
parser.add_argument('--dry_run', action='store_true',default=False, help='Only run on  one file.')
parser.add_argument('--herwig_only', action='store_true',default=False, help='Run on herwig only.')
parser.add_argument('--fine_tuning', action='store_true',default=False, help='Will run data through contrastive space into MLP')
parser.add_argument('--fix_weights', action='store_true',default=False, help='Fixes weights of contrastive embedder')
parser.add_argument('--mpath', action='store', type=str, help='If specified will load model')
parser.add_argument('--abseta', action='store_true',default=False, help='Run on abseta.')
parser.add_argument('--which_augmentations', action='store',type=int,nargs='+',default=None, help='Run on these augmentations (0=seed, 1=fsrUp, 2=fsrDown, 3=herwig7)')
parser.add_argument('--kinematics_only', action='store_true',default=False, help='Train on kinematics only.')
parser.add_argument('--Nmaxsample_train',type = float,help='Number of samples for training')
parser.add_argument('--Nmaxsample_val',type = float,help='Number of samples for validation')
parser.add_argument('--continue_training',action='store_true',default=False,help='Pass this if you want to continue training a prior model. You must also pass mpath')
parser.add_argument('--gvq',action='store_true',default=False,help='Data loaded for quark vs gluon jet classification')
parser.add_argument('--ft_with_p_h',action='store_true',default=False,help='Data loaded for fine tuning with pythia and herwig augmentations')
parser.add_argument('--ft_with_p',action='store_true',default=False,help='Data loaded for fine tuning with pythia nominal only')
parser.add_argument('--is_trans',action='store_true',default=False)
parser.add_argument('--fs_train',action='store_true',default="False",help='Pass this if you are using fully supervised training')
parser.add_argument('--zbb_qcd',action='store_true',default=False,help = 'Pass this if you wish to train Z_bb vs QCD')
parser.add_argument('--top_zbb',action='store_true',default=False,help = 'Pass this if you wish to train top vs Z_bb')
parser.add_argument('--all_augmentations',action='store_true',default=False,help = 'Pass this if you wish to train all augs (used with ft_with_p_h)')
parser.add_argument('--add_reseeded',action='store_true',default=False,)
parser.add_argument('--add_fsr',action='store_true',default=False,)
parser.add_argument('--one_layer_MLP',action='store_true',default=False,)



args = parser.parse_args()
temperature = float(args.temperature)
nepochs = int(args.nepochs)

print(args.ipath)
print("qcd only? ", args.qcd_only)
print("seed only? ", args.seed_only)
print("train with kinematics only? ", args.kinematics_only)
print("train with abseta? ", args.abseta)
print("which augmentations? " , args.which_augmentations)
print("fine tuning? " , args.fine_tuning)

model_dir = args.opath
if not os.path.exists(model_dir):
        os.system("mkdir -p "+model_dir)
        
#Load training data depending on if we are SSL or fully supervised 

if args.zbb_qcd and args.fs_train:
    
    
    data_train = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/outfiles/contrastive_train_QCDonly/', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    qcd_ft_train = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/outfiles/ft_train_QCDonly/', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    zbb_contrastive = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/zzbb-Aug15/contrastive_train/', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    zbb_ft_train = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/zzbb-Aug15/ft_train', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    
    
    
    data_train.data_features = np.concatenate((data_train.data_features,
                                               qcd_ft_train.data_features,
                                              zbb_contrastive.data_features,
                                              zbb_ft_train.data_features))
    data_train.data_truth_label = np.concatenate((data_train.data_truth_label,
                                                  qcd_ft_train.data_truth_label,
                                                 zbb_contrastive.data_truth_label,
                                                 zbb_ft_train.data_truth_label))
    
    data_train.data_jettype = np.concatenate((data_train.data_jettype,
                                             qcd_ft_train.data_jettype,
                                             zbb_contrastive.data_jettype,
                                             zbb_ft_train.data_jettype))
    
    data_train.data_vartype = np.concatenate((data_train.data_vartype,
                                             qcd_ft_train.data_vartype,
                                             zbb_contrastive.data_vartype,
                                             zbb_ft_train.data_vartype))
   
    
    
    num_samples = data_train.data_features.shape[0]

    # Generate a random permutation of indices
    permutation_indices = np.random.permutation(num_samples)
    
    data_train.data_features = data_train.data_features[permutation_indices]
    data_train.data_truth_label = data_train.data_truth_label[permutation_indices]
    data_train.data_jettype = data_train.data_jettype[permutation_indices]
    data_train.data_vartype = data_train.data_vartype[permutation_indices]
    
    
    data_test = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/outfiles/ft_val_QCDonly/', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    v_zbb = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/zzbb-Aug15/ft_val', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    data_test.data_features = np.concatenate((data_test.data_features,v_zbb.data_features))
    data_test.data_truth_label = np.concatenate((data_test.data_truth_label,v_zbb.data_truth_label))
    data_test.data_jettype = np.concatenate((data_test.data_jettype, v_zbb.data_jettype))
    data_test.data_vartype = np.concatenate((data_test.data_vartype, v_zbb.data_vartype))
    
    
    num_samples = data_test.data_features.shape[0]

    # Generate a random permutation of indices
    permutation_indices = np.random.permutation(num_samples)
    
    data_test.data_features = data_test.data_features[permutation_indices]
    data_test.data_truth_label = data_test.data_truth_label[permutation_indices]
    data_test.data_jettype = data_test.data_jettype[permutation_indices]
    data_test.data_vartype = data_test.data_vartype[permutation_indices]

elif args.top_zbb:
    
    
    data_train = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/zvstop/train', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = args.Nmaxsample_train,gvq = args.gvq,args=args)
    
    
    data_test = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/zvstop/test/', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/test", args.dry_run, Nmaxsample = args.Nmaxsample_val,gvq = args.gvq,args=args)
    
    #sys.exit(1)
elif args.zbb_qcd:
    
    
    data_train = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/outfiles/ft_train_QCDonly/', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    


    zbb_ft_train = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/zzbb-Aug15/ft_train', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    
    
    
    data_train.data_features = np.concatenate((data_train.data_features,
                                              zbb_ft_train.data_features))
    data_train.data_truth_label = np.concatenate((data_train.data_truth_label,
                                                 zbb_ft_train.data_truth_label))
    
    data_train.data_jettype = np.concatenate((data_train.data_jettype,
                                             zbb_ft_train.data_jettype))
    
    data_train.data_vartype = np.concatenate((data_train.data_vartype,
                                             zbb_ft_train.data_vartype))
   
    
    
    num_samples = data_train.data_features.shape[0]

    # Generate a random permutation of indices
    permutation_indices = np.random.permutation(num_samples)
    
    data_train.data_features = data_train.data_features[permutation_indices]
    data_train.data_truth_label = data_train.data_truth_label[permutation_indices]
    data_train.data_jettype = data_train.data_jettype[permutation_indices]
    data_train.data_vartype = data_train.data_vartype[permutation_indices]
    
    
    data_test = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/outfiles/ft_val_QCDonly/', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    v_zbb = CLV2_Nate2('/work/tier3/jkrupa/cl/samples/mar20/zzbb-Aug15/ft_val', args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = 2.5e6,gvq = args.gvq,args=args)
    
    data_test.data_features = np.concatenate((data_test.data_features,v_zbb.data_features))
    data_test.data_truth_label = np.concatenate((data_test.data_truth_label,v_zbb.data_truth_label))
    data_test.data_jettype = np.concatenate((data_test.data_jettype, v_zbb.data_jettype))
    data_test.data_vartype = np.concatenate((data_test.data_vartype, v_zbb.data_vartype))
    
    
    num_samples = data_test.data_features.shape[0]

    # Generate a random permutation of indices
    permutation_indices = np.random.permutation(num_samples)
    
    data_test.data_features = data_test.data_features[permutation_indices]
    data_test.data_truth_label = data_test.data_truth_label[permutation_indices]
    data_test.data_jettype = data_test.data_jettype[permutation_indices]
    data_test.data_vartype = data_test.data_vartype[permutation_indices]    
else:
    data_train = CLV2_Nate2(args.ipath, args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/train", args.dry_run, Nmaxsample = args.Nmaxsample_train,gvq = args.gvq,args=args)

    data_test = CLV2_Nate2(args.vpath,args.qcd_only,args.seed_only,args.herwig_only,args.which_augmentations,args.kinematics_only,args.abseta, args.opath+"/test", args.dry_run, Nmaxsample = args.Nmaxsample_val,gvq = args.gvq,args=args)


BATCHSIZE = args.batchsize

train_loader = DataLoader(data_train, batch_size=BATCHSIZE,shuffle=False,
                          follow_batch=['x_pf'])

test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=False,
                         follow_batch=['x_pf'])



import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
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


    """
    SimCLR contrastive loss implemented from the paper A Simple Framework for Contrastive Learning of Visual Representations
    by Chen et al. 
    
    https://arxiv.org/abs/2002.05709
    
    """
def contrastive_loss(x_i, x_j, temperature=0.1):
  
    xdevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 )
    z_j = F.normalize( x_j, dim=1 )
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
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
        self.fine_tuning = args.fine_tuning
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

        )

        if args.one_layer_MLP:
            self.mlp = nn.Sequential(nn.Linear(int(self.n_out_nodes), 1),
                                            nn.Sigmoid()
                                            )
        else:
            self.mlp = nn.Sequential(nn.Linear(int(self.n_out_nodes), 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, 8),
                                            nn.ReLU(),
                                            nn.Linear(8, 1),
                                            nn.Sigmoid()
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
        cspace = out
        if not self.fine_tuning:
            return out, batch
        out = torch.nn.functional.normalize(out,dim=1)
#         print(out[0])
        out = self.mlp(out)
#         print(out[0])
        return out, batch, cspace
            
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training using: {device}')

n_out_nodes = int(args.n_out_nodes)
if args.kinematics_only:
    n_in_nodes=4
else:
    n_in_nodes=15
cl = Net(n_in_nodes,n_out_nodes,args.hidden_dim).to(device)

def load_matching_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)['model']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict,strict=False)

#print(args.mpath != "", args.fs_train, args.continue_training)
#if (args.mpath != "" and args.fs_train) and not args.continue_training:
##    print("Cannot give a model path if you are fully supervising (unless you are continuing to train)")
#    sys.exit()
#Loading Model
if args.continue_training:
    cl.load_state_dict(torch.load(args.mpath)['model'])
    start_epoch = args.mpath.split("/")[-1].split("-")[-1].split(".")[0]
    start_epoch = int(start_epoch) + 1
    print(f"Continuing training from epoch {start_epoch}...")
elif args.mpath:
    load_matching_state_dict(cl,args.mpath)
    print(f'Starting classification with hot start. Loaded model {args.mpath}')
    

#Fix Weights of contrastive embedder
if args.fix_weights:
    print('Freezing Weights')
    for param in cl.parameters():
        param.requires_grad = False
    for name, param in cl.named_parameters():
        #Unfreeze final layers    
        if 'mlp' in name:
          
            param.requires_grad = True

optimizer = torch.optim.Adam(cl.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


def train():
    cl.train()
    counter = 0
    
    total_loss = 0
    for data in tqdm.tqdm(train_loader):
        counter += 1

        data = data.to(device)
        optimizer.zero_grad()
        
        if args.fine_tuning:
            out,_,_ = cl(data.x,data.x_pf_batch)
        else:
            out,_ = cl(data.x,data.x_pf_batch)
        
        if args.fine_tuning:
         
            y = torch.unsqueeze(data.y.float(),dim=1)
            vartype = torch.unsqueeze(data.vartype.float(),dim=1)
            
            
            if args.gvq or args.zbb_qcd or args.top_zbb:
                jet_type = torch.unsqueeze(data.jet_type.float(),dim=1)
                loss = torch.nn.BCELoss()(out,jet_type)
            elif args.ft_with_p_h:
                loss = torch.nn.BCELoss()(out,y)
            else:
                loss = torch.nn.BCELoss()(out,y)
        else:
            loss = contrastive_loss(out[0::2],out[1::2],temperature)

        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
 
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    cl.eval()
    total_loss = 0
    counter = 0
    for data in tqdm.tqdm(test_loader):
        counter += 1
        data = data.to(device)
           
        with torch.no_grad():
            
            if args.fine_tuning:
                out,_,_ = cl(data.x,data.x_pf_batch)
                
                y = torch.unsqueeze(data.y.float(),dim=1)
                vartype = torch.unsqueeze(data.vartype.float(),dim=1)
                if args.gvq or args.zbb_qcd or args.top_zbb:
                    jet_type = torch.unsqueeze(data.jet_type.float(),dim=1)
                    loss = torch.nn.BCELoss()(out,jet_type)
                elif args.ft_with_p_h:
                    loss = torch.nn.BCELoss()(out,y)
                else:
                    loss = torch.nn.BCELoss()(out,y)
                

            else:
                out,_ = cl(data.x_pf,
                       data.x_pf_batch)
                loss = contrastive_loss(out[0::2],out[1::2],temperature)
    
       
            total_loss += loss.item()

       
    return total_loss / len(test_loader.dataset)

    
    
def make_plots(gvq = False):
    cl.eval()
    
    os.system(f'mkdir {args.opath}/plots')
    cspace = []
    truths = []
    num_batches = len(test_loader) - 3
    for batch_idx, data in enumerate(tqdm.tqdm(test_loader)):
        data = data.to(device)
        with torch.no_grad():
            
            if gvq:
                y = data.jet_type
            else:
                y = data.y
            x_pf = x_pf.to(device)
            out = cl(data.x_pf,
                       data.x_pf_batch)
            cspace.append(out[0].cpu())
            truths.append(y.cpu())
    
    data = torch.cat(cspace, dim=0)
    testingLabels = torch.cat(truths,dim=0)
    mask = ~(torch.isnan(data).any(dim=1) | torch.isinf(data).any(dim=1))
    data = data[mask]
    testingLabels = testingLabels[mask]

    data = data.numpy()
    testingLabels = testingLabels.numpy()
    # Find the unique class labels from the testingLabels array
    class_labels = np.unique(testingLabels)

    # Apply PCA to reduce dimensionality to 2 for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    # Create a dictionary to store the data points for each class label
    data_dict = {}
    print(class_labels)
    for label in class_labels:
        data_dict[label] = data_pca[testingLabels == label]

    # Plot the data points
    for label in class_labels:
        plt.scatter(data_dict[label][:, 0], data_dict[label][:, 1], label=f'Class {label}', alpha=0.05)

    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA')
    filename = 'PCA2D'
    plt.savefig(os.path.join(args.opath,'plots', filename))

    # Apply PCA to reduce dimensionality to 3 for 3D visualization
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)
    
    data_dict = {}
    for label in class_labels:
        data_dict[label] = data_pca[testingLabels == label]
        
    # Create a list to store the image frames
    frames = []

    # Create 3D PCA plots at different angles
    for angle in range(0, 360, 5):
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(111, projection='3d')

        for label in class_labels:
            ax.scatter(data_dict[label][:, 0], data_dict[label][:, 1], data_dict[label][:, 2],
                       label=f'Class {label}', alpha=0.05)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D PCA Plot')
        ax.legend()
        ax.grid(False)

        ax.view_init(elev=10, azim=angle)  # Set the elevation and azimuth angles

        # Save the plot as an image file
        filename = f'PCA_Contrastive_Spaces_{angle}.png'
        fig.savefig(os.path.join(args.opath,'plots', filename))
        plt.close(fig)
        time.sleep(1)

        # Open the saved image and append it to the list of frames
        img = Image.open(os.path.join(args.opath,'plots', filename))
        frames.append(img)

    # Save the frames as a GIF
    gif_filename = os.path.join(args.opath,'plots', 'pca_animation.gif')
    frames[0].save(gif_filename, format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=1)



best_val_loss = 1e9

all_train_loss = []
all_val_loss = []


if args.continue_training:
    cut_path = args.mpath.rsplit('/', 2)[0] + '/'
    loss_dict = {'train_loss': pd.read_csv(os.path.join(cut_path,'loss.csv'))['train_loss'].tolist(), 
 'val_loss': pd.read_csv(os.path.join(cut_path,'loss.csv'))['val_loss'].tolist()}
else:
    start_epoch = 1
    loss_dict = {'train_loss': [], 'val_loss': []}

for epoch in range(start_epoch, nepochs):
    print(f'Training Epoch {epoch} on {len(train_loader.dataset)} jets')
    loss = train()
    scheduler.step()
    
    print(f'Validating Epoch {epoch} on {len(test_loader.dataset)} jets')
    loss_val = test()
    
    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    all_train_loss.append(loss)
    all_val_loss.append(loss_val)
    loss_dict['train_loss'].append(loss)
    loss_dict['val_loss'].append(loss_val)
    df = pd.DataFrame.from_dict(loss_dict)
    


    df.to_csv("%s/"%model_dir+"/loss.csv")

    state_dicts = {'model':cl.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}
    if not args.fine_tuning:
        torch.save(state_dicts, os.path.join(model_dir, f'epoch-{epoch}.pt'))

        if loss_val < best_val_loss:
            best_val_loss = loss_val

            torch.save(state_dicts, os.path.join(model_dir, 'best-epoch.pt'.format(epoch)))
    else:
        torch.save(state_dicts, os.path.join(model_dir, f'FT_epoch-{epoch}.pt'))

        if loss_val < best_val_loss:
            best_val_loss = loss_val

            torch.save(state_dicts, os.path.join(model_dir, 'FT_best-epoch.pt'.format(epoch)))

#make_plots(gvq = args.gvq)


