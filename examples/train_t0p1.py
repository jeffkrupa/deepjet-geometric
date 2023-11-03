from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import subprocess
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import h5py
import awkward as ak
import pickle
import pandas as pd
from scipy.spatial import distance_matrix
import sys
tag=sys.argv[1]
with h5py.File(f"./samples/{tag}/qcd_nominal_train.h5", 'r') as f:
    x_train_qcd = f['jet_features'][()]
with h5py.File(f"./samples/{tag}/higgs_nominal_train.h5", 'r') as f:
    x_train_higgs = f['jet_features'][()]
with h5py.File(f"./samples/{tag}/qcd_nominal_val.h5", 'r') as f:
    x_val_qcd = f['jet_features'][()]
with h5py.File(f"./samples/{tag}/higgs_nominal_val.h5", 'r') as f:
    x_val_higgs = f['jet_features'][()]


print(x_train_qcd.shape)
print(x_train_higgs.shape)

min_events = min(x_train_qcd.shape[0],x_train_higgs.shape[0])
x_train_qcd = x_train_qcd[:min_events]
x_train_higgs = x_train_higgs[:min_events]


min_events = min(x_val_qcd.shape[0],x_val_higgs.shape[0])
x_val_qcd = x_val_qcd[:min_events]
x_val_higgs = x_val_higgs[:min_events]


y_train_qcd = np.array([0 for i in range(x_train_qcd.shape[0])])
y_val_qcd = np.array([0 for i in range(x_val_qcd.shape[0])])
y_train_higgs = np.array([1 for i in range(x_train_higgs.shape[0])])
y_val_higgs = np.array([1 for i in range(x_val_higgs.shape[0])])

x_train = ak.Array(np.concatenate((x_train_qcd,x_train_higgs)))
y_train  = ak.Array(np.concatenate((y_train_qcd,y_train_higgs)))
x_val = torch.from_numpy(np.array(ak.Array(np.concatenate((x_val_qcd,x_val_higgs))))).float()
y_val  = torch.from_numpy(np.array(ak.Array(np.concatenate((y_val_qcd,y_val_higgs))))).float()

import random
indices = [i for i in range(len(x_train))]
random.shuffle(indices)

x_train_shuffled = torch.from_numpy(np.array(x_train[indices])).float()
y_train_shuffled = torch.from_numpy(np.array(y_train[indices])).float()



#print(x_train)
#print(x_train_shuffled.shape)
#print(x_train)
#print(y_train_shuffled.shape)





device = "cuda" if torch.cuda.is_available() else "cpu"
#devide = 'cpu'
print(f"Using {device} device")
#torch.cuda.set_device(1)

x_train_shuffled = x_train_shuffled.to(device)
y_train_shuffled = y_train_shuffled.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

model = nn.Sequential(nn.Linear(8, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 32),
                      nn.ReLU(),
                      nn.Linear(32, 8),
                      nn.ReLU(),
                      nn.Linear(8, 1),
                      nn.Sigmoid())

model = model.to(device)

loss_function = nn.BCEWithLogitsLoss()
#loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    pred_y = model(x_train_shuffled)
    #print(pred_y)
    loss = loss_function(torch.flatten(pred_y), y_train_shuffled)
    #losses.append(loss.item())
    #print(loss.item())
    model.zero_grad()
    loss.backward()

    optimizer.step()
    #print(model,x_train_shuffled)
    #sys.exit(1)
    return loss.item()
    
@torch.no_grad()
def test():
    model.eval()
    with torch.no_grad():
        pred_y = model(x_val)
        loss = loss_function(torch.flatten(pred_y), y_val)
        return loss.item()


@torch.no_grad()
def val(x_test):
    model.eval()
    with torch.no_grad():
        pred_y = model(x_test)
        #print(model,x_test)
        return pred_y


def early_stopping(train_loss, validation_loss, min_delta, tolerance):

    counter = 0
    if (validation_loss - train_loss) > min_delta:
        counter +=1
        if counter >= tolerance:
          return True

best_val_loss = 1e9

loss_dict = {'train':[],'val':[]}

last_ten = []
all_losses = [] 
for epoch in range(10000):
    train_loss = train()

    val_loss = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
    epoch, train_loss, val_loss))

    loss_dict['train'].append(train_loss)
    loss_dict['val'].append(val_loss)

    
    #print(pred_y)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        state_dicts = {'model':model.state_dict(),
                       'opt':optimizer.state_dict()} 

        torch.save(state_dicts, f"./samples/{tag}/best-epoch.pt")

    #if early_stopping(loss_dict['train'], loss_dict['val'], min_delta=10, tolerance = 20):
    #  print("We are at epoch:", epoch)
    #  break

    all_losses.append(val_loss)
    if epoch > 60: 
        if np.abs( (np.max(all_losses[epoch-60:epoch-30]) - np.min(all_losses[epoch-60:epoch-30])) ) / np.abs( (np.max(all_losses[epoch-20:epoch]) - np.min(all_losses[epoch-20:epoch])) ) < 1e-2:
            break
        if np.mean(all_losses[epoch-60:epoch-30]) - (np.mean(all_losses[epoch-20:epoch])) < 0:
            break
    #if len(last_ten) > 9:
    #    last_ten.pop(0)
    #    last_ten.append(val_loss)
    #
    #    criterion = (np.max(last_ten) - np.min(last_ten))/np.max(last_ten)
    #    #print(np.max(last_ten))
    #    #print(np.min(last_ten))
    #    #print(criterion)
    #    if criterion < 0.00003:
    #        break
    #else:
    #    last_ten.append(val_loss)

df = pd.DataFrame.from_dict(loss_dict)
df.to_csv(f"./samples/{tag}/losses_t0p1.csv")

#device = "cpu"
for process in ['higgs','qcd']:
    for v in ['nominal','seed','fsrRenHi','fsrRenLo','herwig']:
        if v == 'nominal':
            fname = f'./samples/{tag}/{process}_{v}_val.h5'
        else:
            fname = f'./samples/{tag}/{process}_{v}.h5'
        with h5py.File(fname, 'r') as fi:
            x_test = torch.from_numpy(fi['jet_features'][()]).float().to(device)
            x_kin = fi['jet_kinematics'][()]
            x_jettype = fi['jet_type'][()]
            y = val(x_test)
            #y = val(x_test).cpu().detach().numpy().flatten()
            y = val(x_test).cpu().numpy().flatten()
            #print(fname)
            #print(y)
        dirname = os.path.join(*fname.replace('samples','samples_inferred').split("/")[:-1])
        #print(dirname)
        subprocess.call("mkdir -p %s"%dirname,shell=True)
        hf = h5py.File(fname.replace('samples','samples_inferred'), 'w')
        hf.create_dataset('jet_features', data=y)
        hf.create_dataset('jet_kinematics',data=x_kin)
        hf.create_dataset('jet_type',data=x_jettype)
        hf.close()    
        


'''
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,64,32,8, 1), max_iter=1000,\
                    random_state=1, validation_fraction=0.3, verbose=True, early_stopping=True)
clf.fit(x_train_shuffled,y_train_shuffled)

pickle.dump(clf, open("modelparams.pkl", 'wb'))

clf.predict_proba(x_val_qcd)

'''
