# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import time
import os
import itertools
import shap
from random import choice
from string import ascii_lowercase, digits
from dataclasses import dataclass
import networkx as nx
import pickle
import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, GCNConv, GATv2Conv
from torch.nn import Linear, Dropout
from torch_geometric.utils import add_self_loops, degree, to_networkx
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score,balanced_accuracy_score, precision_recall_curve, auc, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, roc_auc_score, matthews_corrcoef,make_scorer, classification_report
from sklearn.metrics import median_absolute_error, balanced_accuracy_score, precision_score, recall_score, f1_score


# %%
def set_seeds(seed_value, use_cuda):
    random.seed(seed_value)
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
        

set_seeds(551249, True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # Preparing Custom Datasets for Dataloader

# %%
class mm_mgus_dataloader(torch.utils.data.Dataset):

    def __init__(self,sfile,lfile,root_dir):
        
        if isinstance(sfile,str):
            self.name_frame = open(sfile).read().split('\n')[:-1]
            self.label_frame = open(lfile).read().split('\n')[:-1]
        else:
            self.name_frame = sfile
            self.label_frame = lfile
            
        self.root_dir = root_dir

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        
        sname = os.path.join(self.root_dir, self.name_frame[idx])
        feat = pd.read_csv(sname,index_col=0, header=0).to_numpy()
        scaler = StandardScaler()
        feat  =scaler.fit_transform(feat)
        label = torch.tensor(int(self.label_frame[idx])).float()
        sample = (feat,label,self.name_frame[idx])

        return sample

# %% [markdown]
# # Adjacency Matrix Processing

# %%
def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(A.shape[0])
    A_hat = A + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

# %% [markdown]
# # GCN Model Architecture

# %%
adj_mat = pd.read_csv("adj_matrix_798_genes_string_database.csv", index_col=0, header=0)

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        if acti:
            self.acti = nn.LeakyReLU(0.3)
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)

# %%
"""
Model with consensus average of six consensus modules

"""


class GCN(nn.Module):
    
    
    def __init__(self, gcn_parameters, linear_nn_parameters, drop_out, adj_mat_dim):
        super(GCN, self).__init__()
        self.multihead_attn1 = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(adj_mat_dim, adj_mat_dim),
            nn.ReLU(),
            nn.LayerNorm(adj_mat_dim)
        )
        self.multihead_attn2 = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(adj_mat_dim, adj_mat_dim),
            nn.ReLU(),
            nn.LayerNorm(adj_mat_dim)
        )
        
        self.attn_layer = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(2,1),
            nn.ReLU()
        )
        
        self.gcn_layer1 = nn.Sequential(
            GCNLayer(gcn_parameters[0], gcn_parameters[1]),
            nn.Dropout(drop_out)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(linear_nn_parameters[0], linear_nn_parameters[1]),
            nn.LeakyReLU(0.3)
        )
        
    

    def forward(self, X, A, adj_mask = adj_mat):
        adj_mask = torch.from_numpy(adj_mask.to_numpy()).float().reshape(-1,A.shape[0],A.shape[1]).to(device)
        A = torch.from_numpy(preprocess_adj(A)).float().reshape(-1,A.shape[0],A.shape[1]).to(device)
        attn1 = self.multihead_attn1(A).reshape(A.shape[1], A.shape[1])
        attn2 = self.multihead_attn2(A).reshape(A.shape[1], A.shape[1])
        attn_layers = torch.stack((attn1, attn2), dim=1)
        out = []
        for dep in range(attn_layers.shape[0]):
            f = self.attn_layer(torch.transpose(attn_layers[dep,:,:],0,1))
            out.append(torch.transpose(f,0,1))
        x = torch.stack(out,1)
        x = torch.mul(x,adj_mask) #apply masking
        updated_adj_mat = x
        # print(updated_adj_mat.shape, X.shape)
        x = torch.matmul(updated_adj_mat, X.float())
        x = self.gcn_layer1(x)
        x = x.reshape(-1,798)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output, updated_adj_mat

    def compute_l1_loss(self, w):
          return torch.abs(w).sum()

# %% [markdown]
# # Early Stopping

# %%
class EarlyStopping1:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, optimizer, patience=7, verbose=False, delta=0, trace_func=print):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model, perf_mat):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss

# %%
class EarlyStopping2:
    
    def __init__(self, optimizer, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, perf_mat):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        self.val_loss_min = val_loss

# %% [markdown]
# # Training Module

# %%
loss = nn.NLLLoss(weight=torch.from_numpy(np.array([20.,1.])).float()).to(device)

# %%
def train(model, device, train_loader, rand_adj_matrix, optimizer, epoch,log_interval = 10):
    tr_loss = []
    t1 = 0
    t1_with_l1 = 0
    correct = 0
    correctly_predicted_training_samples = []
    correctly_predicted_training_samples_idx = []
    correctly_predicted_training_samples_name = []
    gcn_out = []
    model.train()
    for data, target, sample_names in train_loader:
        
        if torch.is_tensor(rand_adj_matrix):
            rand_adj_matrix = torch.abs(rand_adj_matrix).reshape(798,798)
            rand_adj_matrix = rand_adj_matrix.detach().cpu().numpy()
           
        data = data.float().to(device)
        target =  target.long().to(device)
        optimizer.zero_grad()
        output,rand_adj_matrix_updated = model(data, rand_adj_matrix)
        pred1 = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred1.eq(target.view_as(pred1)).sum().item()
        train_loss = loss(output, target.view(output.shape[0]))

#         Compute L1 loss component
        l1_weight = 0.01
        l1_parameters = []
        for parameter in model.parameters():
            l1_parameters.append(parameter.view(-1))
        l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))

#         Add L1 loss component
        train_loss += l1


        
        t1 += (train_loss-l1).item()
        t1_with_l1 += train_loss.item()
        tr_loss.append(train_loss)

        # print("Inside train original total: ", t1)
        
        train_loss.backward()
        optimizer.step()

    # print('total train loss : ',t1,train_loader.__len__())
    t1 /= train_loader.__len__()
    t1_with_l1  /= train_loader.__len__()
    acc = 100. * correct / len(train_loader.dataset)
#     print('Train Set: Average loss: {:.4f}'.format(t1))    
    return t1, t1_with_l1, acc, rand_adj_matrix_updated

# %% [markdown]
# # Test Module

# %%
def test(model, device, test_loader, rand_adj_matrix, show_perf = True):
    model.eval()
    test_loss = 0
    correct = 0
    target2, pred2 = [], []
    output_pred_prob1 = torch.empty([0]).to(device)
    output_pred_prob = torch.empty([0,2]).to(device)
    correctly_predicted_test_samples = []
    correctly_predicted_test_samples_idx = []
    correctly_predicted_test_samples_name = []
    if torch.is_tensor(rand_adj_matrix):
        rand_adj_matrix = torch.abs(rand_adj_matrix).reshape(798,798)
        rand_adj_matrix = rand_adj_matrix.detach().cpu().numpy()
    model.eval()
    with torch.no_grad():
        for data, target, sample_names in test_loader:
            data = data.to(device).float()
            target = target.long().to(device)
            output,_ = model(data, rand_adj_matrix)
            test_loss += loss(output, target.view(output.shape[0])).item()  # sum up batch loss
            pred1 = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            corerct_idx = torch.where(pred1.eq(target.view_as(pred1)) == True)
            correctly_predicted_test_samples_idx.append(corerct_idx[0])
            correctly_predicted_test_samples.append([data[i] for i in correctly_predicted_test_samples_idx[-1]])
            correctly_predicted_test_samples_name.append([sample_names[i] for i in correctly_predicted_test_samples_idx[-1]])
            correct += pred1.eq(target.view_as(pred1)).sum().item()
            target2.append([i.cpu().item() for i in target])
            pred2.append([j.cpu().item() for j in pred1])
            output_pred_prob1 = torch.cat((output_pred_prob1,torch.max(output, axis=1).values))
            output_pred_prob = torch.cat((output_pred_prob,output))

    target2 = list(itertools.chain.from_iterable(target2))
    pred2 = list(itertools.chain.from_iterable(pred2))
    test_loss /= test_loader.__len__()
    acc = 100. * correct / len(test_loader.dataset)
    cm = confusion_matrix(target2, pred2)   
    cm = {'tn': cm[0, 0], 'fp': cm[0, 1],
          'fn': cm[1, 0], 'tp': cm[1, 1]}
    class_report = classification_report(target2,pred2,output_dict=True)
    f1_score_cal1 = class_report['0']['f1-score']
    prec1 = class_report['0']['precision']
    rec1 = class_report['0']['recall']
    f1_score_cal2 = class_report['1']['f1-score']
    prec2 = class_report['1']['precision']
    rec2 = class_report['1']['recall']
    f1_score_cal = f1_score(target2, pred2)
    f1_score_weighted = f1_score(target2, pred2,average='weighted')
    f1_score_weighted1 = f1_score(target2, pred2,average='weighted',labels=[0])
    f1_score_weighted2 = f1_score(target2, pred2,average='weighted',labels=[1])
    f1_score_micro = f1_score(target2, pred2,average='micro')
    f1_score_macro = f1_score(target2, pred2,average='macro')
    acc = accuracy_score(target2, pred2)
    acc_balanced = balanced_accuracy_score(target2, pred2)
    prec = precision_score(target2, pred2, average = 'weighted')
    rec = recall_score(target2, pred2, average = 'weighted')
    roc = roc_auc_score(target2, output_pred_prob.cpu()[:, 1], average='weighted')
    mcc = matthews_corrcoef(target2, pred2)   
    mgus_prec, mgus_rec,_= precision_recall_curve(target2, output_pred_prob1.cpu().numpy())
    auc2 = auc(mgus_rec,mgus_prec)

    perf_mat = {'accuracy':acc,
                'balanced_accuracy':acc_balanced,
                'f1_score':f1_score_cal,
                'f1_score_MM':f1_score_cal1,
                'f1_score_MGUS':f1_score_cal2,
                'f1_score_weighted':f1_score_weighted,
                'f1_score_weighted_MM':f1_score_weighted1,
                'f1_score_weighted_MGUS':f1_score_weighted2,
                'f1_score_micro':f1_score_micro,
                'f1_score_macro':f1_score_macro,
                'precision':prec,
                'precision_MM':prec1,
                'precision_MGUS':prec2,
                'recall': rec,
                'recall_MM': rec1,
                'recall_MGUS': rec2,
                'roc':roc,
                'mcc': mcc,
                'confusioin_matrix':cm,
                'auc': auc2
               }
    if show_perf:
        print(perf_mat)

    
    return test_loss, acc,perf_mat

# %%
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():        
    
    if hasattr(layer, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()
        if not 'LayerNorm' in str(type(layer)):
          torch.nn.init.zeros_(layer.bias)
          torch.nn.init.xavier_normal_(layer.weight.data) 

# %%
root_dir = '/home/vivek/jupyter_notebooks/bio_dgi_extension/feature_matrix/genes798feature45'
sfile = '/home/vivek/top_model_results/NN/samples_without_PB.txt'
lfile = '/home/vivek/top_model_results/NN/labels_without_PB_new.txt'
X = open(sfile).read().split('\n')
Y = open(lfile).read().split('\n')
# Shuffling the data
X1,Y1 =[], []
new_idx = list(np.arange(len(X)))
new_idx = random.sample(new_idx,len(new_idx))
for i in new_idx:
    X1.append(X[i])
    Y1.append(Y[i])

def load_folds(fold_no):
    if fold_no == 1:
        x_train = open('new_folds/xtrain_fold1.txt').read().split('\n')
        y_train = open('new_folds/ytrain_fold1.txt').read().split('\n')
        x_test = open('new_folds/xtest_fold1.txt').read().split('\n')
        y_test = open('new_folds/ytest_fold1.txt').read().split('\n')

    elif fold_no == 2:
        x_train = open('new_folds/xtrain_fold2.txt').read().split('\n')
        y_train = open('new_folds/ytrain_fold2.txt').read().split('\n')
        x_test = open('new_folds/xtest_fold2.txt').read().split('\n')
        y_test = open('new_folds/ytest_fold2.txt').read().split('\n')

    elif fold_no == 3:
        x_train = open('new_folds/xtrain_fold3.txt').read().split('\n')
        y_train = open('new_folds/ytrain_fold3.txt').read().split('\n')
        x_test = open('new_folds/xtest_fold3.txt').read().split('\n')
        y_test = open('new_folds/ytest_fold3.txt').read().split('\n')

    elif fold_no == 4:
        x_train = open('new_folds/xtrain_fold4.txt').read().split('\n')
        y_train = open('new_folds/ytrain_fold4.txt').read().split('\n')
        x_test = open('new_folds/xtest_fold4.txt').read().split('\n')
        y_test = open('new_folds/ytest_fold4.txt').read().split('\n')

    elif fold_no == 5:
        x_train = open('new_folds/xtrain_fold5.txt').read().split('\n')
        y_train = open('new_folds/ytrain_fold5.txt').read().split('\n')
        x_test = open('new_folds/xtest_fold5.txt').read().split('\n')
        y_test = open('new_folds/ytest_fold5.txt').read().split('\n')

    return x_train, x_test, y_train, y_test


# %%
epochs = 1000
fold_no = 1
batch_size = 64
patience = 35
overall_tp,overall_fp,overall_tn,overall_fn = [],[],[],[]
train_loss, test_loss, test_acc, score, final_cm = {},{},{}, {}, {}
acc,balanced_acc,f1_sc,f1_score_weighted, f1_score_micro,f1_score_macro = [],[],[],[],[],[]
prec,rec,sav,roc,mcc = [],[],[],[],[]
prec_mgus,prec_mm,f1_mgus,f1_mm = [],[],[],[]
f1_wt_mgus,f1_wt_mm = [],[]
rec_mgus,rec_mm = [],[]
acc_train, acc_balanced_train = [], []
output_pred_prob2 = []
train_loss_with_l1, change_adj_matrix_dict = {}, {}
modularity_dict, graph_energy_dict = {}, {}
corr_pred_train_sample_dict, corr_pred_train_sample_name_dict = {}, {}
corr_pred_test_sample_dict, corr_pred_test_sample_name_dict = {}, {}
recall, precision = {}, {}
kf = StratifiedKFold(n_splits=5, random_state = 108, shuffle=True)
gcn_out_dict = {}
shap_values = {}
rand_adj_matrix_dict = {}
gene_dict = {}
gene_list = ['KRAS','NRAS', 'BRAF', 'MAX', 'IGLL5'] #List of inportant genes
train_acc, test_acc = dict(), dict()


for fold_no in range(1,6):
    rand_adj_matrix_dict[fold_no] = dict()
    print(f'**********************Fold-{fold_no} Training*************************')
    model = GCN((45,1),(798,2),0.75,798).to(device)
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=1e-5)
    early_stopping1 = EarlyStopping1(optimizer,
                                   patience=patience,
                                   trace_func=warnings.warn)
    early_stopping2 = EarlyStopping2(optimizer,
                                   patience=patience,
                                   path='/home/vivek/jupyter_notebooks/bio_dgi_extension/ppi_string/model2/GCN_model_'+str(fold_no)+'_checkpoint.pt',
                                   trace_func=warnings.warn) #warnings.warn
    test_loss['fold_'+str(fold_no)] = []
    train_loss['fold_'+str(fold_no)] = []
    train_loss_with_l1['fold_'+str(fold_no)] = []
    test_acc['fold_'+str(fold_no)] = []
    train_acc['fold_'+str(fold_no)] = []
    change_adj_matrix_dict['fold_'+str(fold_no)] =[]
    if os.path.exists('/home/vivek/jupyter_notebooks/bio_dgi_extension/ppi_string/model2/learned_adj_mat/fold_'+str(fold_no)+'/adj_mat_0.csv'):
        os.system('rm /home/vivek/jupyter_notebooks/bio_dgi_extension/ppi_string/model2/learned_adj_mat/fold_'+str(fold_no)+'/adj_mat*.csv')
    X_train, X_test, y_train, y_test = load_folds(fold_no)
    tr_mm_mgus_dataset = mm_mgus_dataloader(sfile = X_train, lfile = y_train, root_dir = root_dir)
    val_mm_mgus_dataset = mm_mgus_dataloader(sfile = X_test, lfile = y_test, root_dir = root_dir)
    train_loader = DataLoader(tr_mm_mgus_dataset,batch_size=batch_size,shuffle=False, num_workers=25)
    val_loader = DataLoader(val_mm_mgus_dataset,batch_size=batch_size,shuffle=False, num_workers=25)
    adj_mat = pd.read_csv("/home/vivek/jupyter_notebooks/bio_dgi_extension/adj_mats/adj_matrix_798_genes_string_database.csv", index_col=0, header=0)

    for iterations in tqdm(range(epochs)):
        tr, tr_with_l1, tr_acc, learned_adj_matrix = train(model, device, train_loader, adj_mat, optimizer, iterations)
        df = pd.DataFrame(learned_adj_matrix.reshape(798,798).cpu().detach().numpy(), columns=adj_mat.columns, index=adj_mat.index)
        df.to_csv(os.path.join('/home/vivek/jupyter_notebooks/bio_dgi_extension/ppi_string/model2/learned_adj_mat','fold_'+str(fold_no),'adj_mat_'+str(iterations)+'.csv'), index = True, encoding='utf-8')
        rand_adj_matrix_dict[fold_no][iterations] = learned_adj_matrix
        if iterations == epochs-1:
            te, te_acc,perf_mat = test(model, device, val_loader, learned_adj_matrix, show_perf=True)
        else:
            te, te_acc,perf_mat = test(model, device, val_loader, learned_adj_matrix, show_perf=False)

        train_loss['fold_'+str(fold_no)].append(tr)
        train_loss_with_l1['fold_'+str(fold_no)].append(tr_with_l1)
        test_loss['fold_'+str(fold_no)].append(te)
        train_acc['fold_'+str(fold_no)].append(tr_acc)
        test_acc['fold_'+str(fold_no)].append(te_acc)        
        early_stopping1(te, model, perf_mat)

        if iterations >= 1:
            change_adj_matrix = torch.norm(rand_adj_matrix_dict[fold_no][iterations] - rand_adj_matrix_dict[fold_no][iterations-1], 2)
            change_adj_matrix_dict['fold_'+str(fold_no)].append(change_adj_matrix)
            early_stopping2(change_adj_matrix_dict['fold_'+str(fold_no)][-1], model, perf_mat)
            # writer.add_scalars('change_adj_matrix_fold' + str(fold_no), {'change_adj_matrix':change_adj_matrix}, global_step=iterations)

        # writer.add_scalars('train_val_Loss_fold' + str(fold_no), {'Test_loss':te}, global_step=iterations)
        # writer.add_scalars('train_val_Loss_fold' + str(fold_no), {'Train_loss':tr}, global_step=iterations)
        # writer.add_scalars('train_val_Loss_fold' + str(fold_no), {'Train_loss_L1':tr_with_l1}, global_step=iterations)
        # writer.add_scalars('Modularity_index_fold' + str(fold_no), {'modularity':modularity}, global_step=iterations)
        # writer.add_scalars('Graph_energy_fold' + str(fold_no), {'graph_energy':graph_energy}, global_step=iterations)


        if early_stopping1.early_stop and early_stopping2.early_stop:
            print(perf_mat['confusioin_matrix'])
            print("############# Early stopping ###############")
            break



    overall_tp.append(perf_mat['confusioin_matrix']['tp'])
    overall_tn.append(perf_mat['confusioin_matrix']['tn'])
    overall_fp.append(perf_mat['confusioin_matrix']['fp'])
    overall_fn.append(perf_mat['confusioin_matrix']['fn'])
    acc.append(perf_mat['accuracy'])
    balanced_acc.append(perf_mat['balanced_accuracy'])
    f1_sc.append(perf_mat['f1_score'])
    f1_mgus.append(perf_mat['f1_score_MGUS'])
    f1_mm.append(perf_mat['f1_score_MM'])
    f1_score_weighted.append(perf_mat['f1_score_weighted'])
    f1_wt_mgus.append(perf_mat['f1_score_weighted_MGUS'])
    f1_wt_mm.append(perf_mat['f1_score_weighted_MM'])
    f1_score_micro.append(perf_mat['f1_score_macro'])
    f1_score_macro.append(perf_mat['f1_score_micro'])
    prec.append(perf_mat['precision'])
    prec_mgus.append(perf_mat['precision_MGUS'])
    prec_mm.append(perf_mat['precision_MM'])
    rec.append(perf_mat['recall'])
    rec_mgus.append(perf_mat['recall_MGUS'])
    rec_mm.append(perf_mat['recall_MM'])
    roc.append(perf_mat['roc'])
    mcc.append(perf_mat['mcc'])
    del model
    torch.cuda.empty_cache()
    fold_no += 1

final_cm['tp'] = sum(overall_tp)
final_cm['fp'] = sum(overall_fp)
final_cm['tn'] = sum(overall_tn)
final_cm['fn'] = sum(overall_fn)
print('************************************')
print('The final confusion matrix is : ',final_cm)

# %%
# Model Performance analysis
print('The mean accuracy is                  :',np.mean(acc))
print('The mean balanced accuracy is         :',np.mean(balanced_acc))
print('The mean f1-score is                  :',np.mean(f1_sc))
print('The mean f1-score for MGUS is         :',np.mean(f1_mgus))
print('The mean f1-score for MM is           :',np.mean(f1_mm))
print('The mean weighted f1-score is         :',np.mean(f1_score_weighted))
print('The mean weighted f1-score for MGUS is:',np.mean(f1_wt_mgus))
print('The mean weighted f1-score for MM is  :',np.mean(f1_wt_mm))
print('The mean micro f1-score is            :',np.mean(f1_score_micro))
print('The mean macro f1-score is            :',np.mean(f1_score_macro))
print('The mean precision is                 :',np.mean(prec))
print('The mean precision for MGUS is        :',np.mean(prec_mgus))
print('The mean precision for MM is          :',np.mean(prec_mm))
print('The mean recall for MGUS is           :',np.mean(rec_mgus))
print('The mean recall for MM is             :',np.mean(rec_mm))
print('The mean recall is                    :',np.mean(rec))
print('The mean MCC is                       :',np.mean(mcc))
print('The mean saving score is              :',np.mean(sav))
print('The mean ROC is                       :',np.mean(roc))

# %%
change_adj_matrix_dict2 = {}
change_adj_matrix_dict2['fold_1'] = [i.cpu().detach().tolist() for i in change_adj_matrix_dict['fold_1']]
change_adj_matrix_dict2['fold_2'] = [i.cpu().detach().tolist() for i in change_adj_matrix_dict['fold_2']]
change_adj_matrix_dict2['fold_3'] = [i.cpu().detach().tolist() for i in change_adj_matrix_dict['fold_3']]
change_adj_matrix_dict2['fold_4'] = [i.cpu().detach().tolist() for i in change_adj_matrix_dict['fold_4']]
change_adj_matrix_dict2['fold_5'] = [i.cpu().detach().tolist() for i in change_adj_matrix_dict['fold_5']]
# %%
with open('train_loss.pickle', 'wb') as handle:
    pickle.dump(train_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('validation_loss.pickle', 'wb') as handle:
    pickle.dump(test_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_acc.pickle', 'wb') as handle:
    pickle.dump(test_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)    

with open('train_acc.pickle', 'wb') as handle:
    pickle.dump(train_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)    

with open('cha_adj_mat.pickle', 'wb') as handle:
    pickle.dump(change_adj_matrix_dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('graph_energy.pickle', 'wb') as handle:
#     pickle.dump(graph_energy_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
torch.cuda.empty_cache()