
import schnetpack.atomistic.output_modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

import schnetpack as spk
import schnetpack.representation as rep
from schnetpack.datasets import *
def calc_lr2(lr, decay):
    return lr/(1 + decay)
lr = 2.0e-4
import sys
from schnetpack.train import build_mse_loss
#python dev.py 0 0 1 1 u .1 True
softmax = int(sys.argv[1])
division = int(sys.argv[2])
layers_w = int(sys.argv[3])
layers_c = int(sys.argv[4])
train_test = sys.argv[5]
dropout = float(sys.argv[6])
exp = sys.argv[7]
if exp is not None:
    exp = True
else:
    exp = False
exp = False
def DataByCount(QM9t, n_idx, mm = [0,10],max_idx = 50000):
    """
    input: dataset, num_datapoints, min and max heavy atoms[min,max]
    """
    idx = []
    n_count = 0
    c_idx = 0
    while n_count < n_idx and max_idx > c_idx:
        Z = QM9t.get_properties(c_idx)[1]['_atomic_numbers']


        count = 0
        for j in Z:
            if j > 1:
                count +=1
        if count >= mm[0] and count <=mm[1]:
            idx.append(c_idx)
            n_count +=1
        c_idx +=1
        if n_count >= n_idx:
            break
    return idx

# load qm9 dataset and download if necessary
data = QM9("src/examples/qm9.db")
train_n = 10000
val_n = 10
# split in train and val
train, val, test = data.create_splits(train_n, val_n)
if train_test == 'u':
    train = data.create_subset(DataByCount(data,train_n,mm=[5,7]))
    val = data.create_subset(DataByCount(data,val_n,mm=[8,9]))
    val2 = data.create_subset(DataByCount(data,val_n,mm=[5,7]))

elif train_test =='d':
    train = data.create_subset(DataByCount(data,train_n,mm=[8,9]))
    val = data.create_subset(DataByCount(data,val_n,mm=[5,7]))
    val2 = data.create_subset(DataByCount(data,val_n,mm=[8,9]))

else:
    train = data.create_subset(DataByCount(data,train_n,mm=[5,9]))
    val = data.create_subset(DataByCount(data,val_n,mm=[5,9]))
    train_test = 'a'
loader = spk.data.AtomsLoader(train, batch_size=500, num_workers=4)
val_loader = spk.data.AtomsLoader(val)
val2_loader = spk.data.AtomsLoader(val2)

properties = ["energy", "forces"]  # properties used for training

# create model
#---------NEW----------------#
#pick how many attention heads will be in the interaction convolution (conv) or the
#weight generating scheme (weights)
#NOTE: for now, n_heads_conv must be equal to 1 (the same size as the generated weight matrix)
#NOTE: Other possibilities exist to increase this size if n_heads_conv > 1, or to ouput the dimension of heads
#NOTE: to be smaller than dimension of weights.


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
print('device is %s\n'%dev)
#hyperparameters correspond to a softmax in the attention layers, and also
#normalization by the dimension of the square root of the dimension of attention
#NOTE:during testing on ethanol, energy predictions were lower without softmax and division
#setting to 0 means the attention network does not use this hyperparameter
#setting to 1 means the hyperparater is used
hyperparams = [softmax,division] #[softmax,division by sqrt(dim_k)
print(dropout, 'dropout')
# layers_w, layers_c
reps = rep.SchNet(n_heads_weights=layers_w,n_heads_conv=layers_c,device = torch.device(dev),\
        hyperparams=hyperparams,dropout = dropout,exp=exp,n_interactions=3, normalize_filter = True)
# -----------------------#

#
output = schnetpack.atomistic.Atomwise(n_in=reps.n_atom_basis,aggregation_mode='avg')
model = schnetpack.atomistic.AtomisticModel(reps, output)

loss = lambda b, p: F.l1_loss(p["y"], b[QM9.U0])#mse_loss, l1_loss
numepoch = 10000
numepochepoch = 10
for epoch in range(numepoch):
#    if epoch %2 ==0:
#        v = val_loader
#        vp = 'primary'
#    else:
#        v = val2_loader
#        vp = 'secondary'
    v = val_loader
    a = (epoch)/numepoch
    lr = calc_lr2(lr, 1.0e-4)
    opt = Adam(model.parameters(), lr=lr)
    trainer = spk.train.Trainer("output%s%s%s%s%s/"%(softmax,division,layers_w,layers_c,train_test), model, loss, opt, loader, v)

    # start training
    print(':%s'%epoch)
    trainer.train(torch.device(dev),n_epochs=numepochepoch)
