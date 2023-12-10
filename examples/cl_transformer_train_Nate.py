import sklearn
import numpy as np
from random import randrange
import subprocess
import tqdm
import pandas as pd
from transformers.models.bert.modeling_bert import ACT2FN, BertEmbeddings, BertSelfAttention, prune_linear_layer
from transformers.activations import gelu_new
#from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
#from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv

from deepjet_geometric.datasets import CLV1_Torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import utils
from torch.nn import CrossEntropyLoss, MSELoss
import math
import yaml 
from yaml.loader import SafeLoader
import os, sys
import argparse
import json 
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from PIL import Image
import os
import time

import sys
#sys.path.append('HypJet')
#from HypJet.hyptorch import nn as hypnn
#from HypJet.hyptorch import pmath



world_size = 4  # Number of GPUs you want to use 


BATCHSIZE = 200
VERBOSE = False

p = utils.ArgumentParser()
p.add_args(
    '--dataset_pattern', '--output', ('--n_epochs', p.INT),
    '--checkpoint_path',
    ('--ipath', p.STR), ('--vpath', p.STR), ('--opath', p.STR),
    ('--temperature', p.FLOAT), ('--n_out_nodes',p.INT), ('--n_max_train',p.INT), ('--n_max_val',p.INT),
    ('--qcd_only',p.STORE_TRUE), ('--seed_only',p.STORE_TRUE),
    ('--abseta',p.STORE_TRUE), ('--kinematics_only',p.STORE_TRUE),
    ('--istransformer',p.STORE_TRUE),
    ('--num_encoders', p.INT),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batch_size', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--attention_band', p.INT),
    ('--epoch_offset', p.INT),
    ('--from_snapshot'),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--pt_weight', p.STORE_TRUE), ('--num_max_files', p.INT),
    ('--num_max_particles', p.INT), ('--dr_adj', p.FLOAT),
    ('--beta', p.STORE_TRUE),('--hyperbolic', p.STORE_TRUE),('--c', p.FLOAT),
    ('--lr_policy'), ('--grad_acc', p.INT), ('--is_decoder',p.STORE_TRUE), ('--replace_mean',p.STORE_TRUE)
)
config = p.parse_args()

os.system(f"mkdir -p {config.opath}")

stdoutOrigin=sys.stdout 
sys.stdout = open("./"+config.opath+"/log.txt", "w")
#json_object = json.dumps(config.toJSON(), indent=4)
#with open("./"+config.opath+"/config.json", "w") as outfile:
#    outfile.write(json_object)
config.save_to("./"+config.opath+"/config.yaml")

with open(f"./{config.opath}/config.json", "w") as outfile:
    json.dump(config.__dict__, outfile, indent=4)
    
#sys.exit(1) 
print(config)
sys.stdout.close()
sys.stdout=stdoutOrigin

#parser = argparse.ArgumentParser(description='Test.')
#parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
#parser.add_argument('--vpath', action='store', type=str, help='Path to validation files.')
#parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
#parser.add_argument('--temperature', action='store', type=str, help='SIMCLR Temperature.')
#parser.add_argument('--nepochs', action='store', type=str, help='Number of epochs to train for.')
#parser.add_argument('--n_out_nodes', action='store', type=int, help='Number of output (encoded) nodes.')
#parser.add_argument('--qcd_only', action='store_true',default=False, help='Run on QCD only.')
#parser.add_argument('--seed_only', action='store_true',default=False, help='Run on seed only.')
#parser.add_argument('--abseta', action='store_true',default=False, help='Run on abseta.')
#parser.add_argument('--kinematics_only', action='store_true',default=False, help='Train on kinematics only.')


#args = parser.parse_args()
temperature = float(config.temperature)
nepochs = int(config.n_epochs)
print(config.ipath)
print("qcd only? ", config.qcd_only)
print("seed only? ", config.seed_only)
print("train with kinematics only? ", config.kinematics_only)
print("train with abseta? ", config.abseta)
model_dir = config.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)
    #subprocess.call("mkdir -p %s"%model_dir,shell=True)
data_train = CLV1_Torch(config.ipath, Nevents = config.n_max_train, opath = config.opath+"/train")
data_test = CLV1_Torch(config.vpath, Nevents = config.n_max_val, opath=  config.opath+"/test")


train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, num_replicas=world_size, rank=rank)
test_sampler = torch.utils.data.distributed.DistributedSampler(data_train, num_replicas=world_size, rank=rank)


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, sampler=train_sampler, num_workers=16)
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, sampler=test_sampler, num_workers=16)

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_scatter import scatter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl', # 'nccl' is recommended for GPU, 'gloo' for CPU
        init_method='env://', # Uses environment variables for configuration
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

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
    #print("x_i",x_i)
    #print("x_j",x_j)
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
    #print(nominator, denominator)
    loss_partial = -torch.log( nominator / torch.sum( denominator, dim=1 ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss


class OskarAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()
        self.attention_band = config.attention_band

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.attention_band is not None:
            query_layer = query_layer.permute(0, 2, 1, 3)
            key_layer = key_layer.permute(0, 2, 1, 3)
            value_layer = value_layer.permute(0, 2, 1, 3)

            attn_band = self.attention_band 
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
                remove_from_windowed_attention_mask = (attention_mask != 0)
            query_layer /= math.sqrt(self.attention_head_size)
            query_layer = query_layer.float().contiguous() 
            key_layer = key_layer.float().contiguous() 
            if False:
                attention_scores = diagonaled_mm_tvm(
                        query_layer, key_layer,
                        attn_band, 
                        1, False, 0, False # dilation, is_t1_diag, padding, autoregressive
                    )
            else:
                attention_scores = sliding_chunks_matmul_qk(
                        query_layer, key_layer,
                        attn_band, padding_value=0
                )
            mask_invalid_locations(attention_scores, attn_band, 1, False)
            if attention_mask is not None:
                remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                float_mask = remove_from_windowed_attention_mask.type_as(query_layer).masked_fill(remove_from_windowed_attention_mask, -10000.0)
                float_mask = float_mask.repeat(1, 1, 1, 1) # don't think I need this
                ones = float_mask.new_ones(size=float_mask.size())  
                if False:
                    d_mask = diagonaled_mm_tvm(ones, float_mask, attn_band, 1, False, 0, False)
                else:
                    d_mask = sliding_chunks_matmul_qk(ones, float_mask, attn_band, padding_value=0)
                attention_scores += d_mask

            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
            attention_probs = self.dropout(attention_probs)
            
            value_layer = value_layer.float().contiguous()
            if False:
                context_layer = diagonaled_mm_tvm(attention_probs, value_layer, attn_band, 1, True, 0, False)
            else:
                context_layer = sliding_chunks_matmul_pv(attention_probs, value_layer, attn_band)

        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if VERBOSE:
                # print(attention_probs[0, :8, :8])
                print(torch.max(attention_probs), torch.min(attention_probs))

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3)

        context_layer = context_layer.contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class OskarLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = OskarAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        try:
            self.activation = ACT2FN[config.hidden_act]
        except KeyError:
            self.activation = config.hidden_act

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
#         print(hidden_states.shape)
#         print(self.attention)
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class OskarLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([OskarLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class OskarTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([OskarLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
    
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(config)
        self.relu = gelu_new #nn.ReLU() 
        self.tanh = nn.Tanh()
        self.c = config.c

        config.output_attentions = False
        config.output_hidden_states = False
        config.num_hidden_groups = 1
        config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"

        self.input_bn = nn.BatchNorm1d(config.feature_size) 

        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        self.embed_bn = nn.BatchNorm1d(config.embedding_size) 

        self.encoders = nn.ModuleList([OskarTransformer(config) for _ in range(config.num_encoders)])
        self.decoders = nn.ModuleList([
                                       nn.Linear(config.hidden_size, config.hidden_size), 
                                       nn.Linear(config.hidden_size, config.hidden_size), 
                                       nn.Linear(config.hidden_size, config.n_out_nodes)
                                       ])
        self.decoder_bn = nn.ModuleList([nn.BatchNorm1d(config.hidden_size) for _ in self.decoders[:-1]])
        #self.pooling = torch.mean()
        self.tests = nn.ModuleList(
                    [
                      nn.Linear(config.feature_size, 1, bias=False),
                      # nn.Linear(config.feature_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, 1)
                    ]
                    )

        self.final_embedder = nn.ModuleList([
            nn.Linear(config.n_out_nodes, config.n_out_nodes),
            nn.Linear(config.n_out_nodes, config.n_out_nodes),
        ])

        self.config = config
        size = 100
       
        if config.replace_mean:    
            self.pre_final = nn.ModuleList([
                                           nn.Linear(int(size), int(size/2)),
                                           nn.Linear(int(size/2), int(size/4)),
                                           nn.Linear(int(size/4), int(1))
                                           ])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask=None):
        
        if mask is None:
            mask = torch.ones(x.size()[:-1], device=device)
        if len(mask.shape) == 3:
            attn_mask = mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
        else:
            attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]
        attn_mask = (1 - attn_mask) * -1e9

        head_mask = [None] * self.config.num_hidden_layers

        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        h = self.embedder(x) 
        h = torch.relu(h)
        h = self.embed_bn(h.permute(0, 2, 1)).permute(0, 2, 1)
#         print(h.shape)
        for e in self.encoders:
            h = e(h, attn_mask, head_mask)[0]
        h = self.decoders[0](h)
        h = self.relu(h)
        h = self.decoder_bn[0](h.permute(0, 2, 1)).permute(0, 2, 1)

        h = self.decoders[1](h)
        h = self.relu(h)
        h = self.decoder_bn[1](h.permute(0, 2, 1)).permute(0, 2, 1)

        h = self.decoders[2](h)   # [Batch size, #particles, #output features]
        if self.config.replace_mean:
#             print('in foward')
#             print(h.shape)
            h = torch.reshape(h,(h.shape[0],h.shape[2],h.shape[1]))
#             print(h.shape)
#             print(self.pre_final)
            h = self.pre_final[0](h)
            h = self.pre_final[1](h)
            h = self.pre_final[2](h)
            h = torch.squeeze(h,dim =2)
        else:
            h = torch.mean(h,dim=1)   # [Batch size, #output features]
        
        h = self.final_embedder[0](h)
        h = self.relu(h)
        h = self.final_embedder[1](h)
        h = self.relu(h)
        
        return h

        

n_out_nodes = int(config.n_out_nodes)

cl = Transformer(config)
cl = cl.to(rank)
cl = DDP(cl,device_ids=[rank])

optimizer = torch.optim.Adam(cl.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


#def train():
def train(rank, world_size):
    setup(rank, world_size)

    cl.train()
    counter = 0

    total_loss = 0
    num_batches = len(train_loader) +1
    for batch_idx, data in enumerate(tqdm.tqdm(train_loader)):
        counter += 1
        if batch_idx >= num_batches:
            break

        data = data.to(device)
        optimizer.zero_grad()
     
        cur_batchsize = min(BATCHSIZE, data.x_pf.shape[0]/100)
        x_pf = torch.reshape(data.x_pf,(int(cur_batchsize),100,15))
        
        x_pf = x_pf.to(device)
        out = cl(x_pf)
        loss = contrastive_loss(out[0::2],out[1::2],temperature)
        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    cleanup()   
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    cl.eval()
    
    total_loss = 0
    counter = 0
    num_batches = len(test_loader) - 3
    for batch_idx, data in enumerate(tqdm.tqdm(test_loader)):
        if batch_idx >= num_batches:
            break
        counter += 1
        data = data.to(device)
        with torch.no_grad():
            cur_batchsize = min(BATCHSIZE, data.x_pf.shape[0]/100)
            x_pf = torch.reshape(data.x_pf,(int(cur_batchsize),100,15))
            x_pf = x_pf.to(device)
            out = cl(x_pf)
            

            loss = contrastive_loss(out[0::2],out[1::2],temperature)

       
            total_loss += loss.item()

    
    
    
    return total_loss / len(test_loader.dataset)
    

best_val_loss = 1e9

all_train_loss = []
all_val_loss = []

loss_dict = {'train_loss': [], 'val_loss': []}

for epoch in range(1, nepochs):
    print(f'Training Epoch {epoch} on {len(train_loader.dataset)} jets')
    #loss = train()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
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
    
    

    df.to_csv("%s/"%model_dir+"/loss.csv")
    
    state_dicts = {'model':cl.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}

    torch.save(state_dicts, os.path.join(model_dir, f'epoch-{epoch}.pt'))

    if loss_val < best_val_loss:
        best_val_loss = loss_val

        torch.save(state_dicts, os.path.join(model_dir, 'best-epoch.pt'.format(epoch)))

print(all_train_loss)
print(all_val_loss)

