import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from schnetpack.nn.initializers import zeros_initializer


__all__ = ["Dense", "GetItem", "ScaleShift", "Standardize", "Aggregate","AttentionNetwork","AttentionHeads"]

#attention module for cfconv
class AttentionHeads(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        atomic_embedding_dim,
        n_heads=2,
        EXP = False,
        SM = False,
        IN_WEIGHTS = False,
        device = torch.device("cpu"),
        hyperparams = [0,0],
        dropout = 0
    ):
        super(AttentionHeads,self).__init__()
        self.device = device
        self.hp = hyperparams
        self.in_features = in_features
        self.out_features = out_features #output dim of each attention head
        self.atomic_embedding_dim = atomic_embedding_dim #output of linear layers after cat(attention_heads)
        self.n_heads = n_heads
        self.EXP = False
        self.SM = SM
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleList([AttentionNetwork(self.in_features, \
            self.out_features,self.out_features,EXP=self.EXP,device=self.device,SM = self.SM,\
            hyperparams = self.hp) for _ in range(n_heads)])
        self.batchnormheads = nn.BatchNorm1d(self.n_heads*self.out_features)
        self.final = nn.Linear(self.n_heads*self.out_features, self.atomic_embedding_dim)####
        self.batchnormfinal =  nn.BatchNorm1d(self.atomic_embedding_dim)
        self.output = []
    def forward(self,x,Weights = None ):
        shape = x.shape
        self.output = []
        x = self.dropout(x)
        for i in range(self.n_heads):
            if Weights is not None:
                self.output.append(self.heads[i](x,Weights))
            else:
                self.output.append(self.heads[i](x))

        f = self.dropout(self.final(self.batchnormheads(self.dropout(torch.cat(self.output,dim=3))\
            .view(-1,self.out_features*self.n_heads))))
        f = self.batchnormfinal(f)
        return f.view(-1,shape[1],shape[1]-1,self.atomic_embedding_dim)

class AttentionNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        EXP = False,
        SM = False,
        device = torch.device("cpu"),
        hyperparams = [0,0]
            ):
        super(AttentionNetwork, self).__init__()
        self.device = device
        self.hp = hyperparams
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.EXP = EXP
        self.SM = SM
        self.softmax = nn.Softmax(dim=1)
        ###
        #------------  maybe only linear layer, not non linear
        ###
        self.AttentionQuery = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features)
            )
        self.AttentionValue = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features)
            )
        self.AttentionVector = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features)
            )
    def GetPairs(self,embeddings):
        #embedding should have dimensions(batch, L, L, dim(embedding))
        shape = embeddings.shape
        embeddings = embeddings.repeat(1,shape[1],1).view(shape[0],shape[1],shape[1],-1)
        embeddingsT = embeddings.transpose(2,1)
        return embeddings, embeddingsT
    def forward(self, AtomEmbeddings,Weights=None,mask=False,gather = True):
        #Atom_embeddings shape [batch, n_atoms, n_atoms-1, dim(embedding)]
        querys, values = self.GetPairs(AtomEmbeddings)
        shape = querys.shape

        #split atom pairs
        querys = querys.reshape(-1,shape[-1])
        values = values.reshape(-1,shape[-1])

        Q = self.AttentionQuery(querys).reshape(-1,self.out_features)
        V = self.AttentionValue(values).reshape(-1,self.out_features)
        # calculate scalar attention
        a = torch.einsum('bi,bj->b',Q, V).reshape(shape[0],shape[1],shape[1])
        # hyperparameters

        #if you should divide scalar attention values by sqrt(dimension(Q))
        if self.hp[1] ==1:
            a /= torch.sqrt(torch.FloatTensor([self.out_features])).to(self.device) #divide by dimension of Q
        #if exponential should be used
        if self.EXP is True:
            a = torch.exp(a)
        #if softmax should be used, like in transformer networks
        if self.SM is True or self.hp[0] ==1:
            a = self.softmax(a)
        #calculate vector attention
        if Weights is None:
            b = self.AttentionVector(values).reshape(shape[0],shape[1],shape[1],-1)
        else: #weights instead of values
            #concat diagonal to make L by L matrix
            b = self.AttentionVector(Weights.view(-1,shape[-1])).reshape(shape[0],shape[1],shape[1]-1,-1)
        if mask is False and gather is False:
            return torch.einsum('ijkl, ijk -> ijkl',b,a)
        elif gather is True:
            if Weights is None:
                b = torch.einsum('ijkl, ijk -> ijkl',b,a).to(self.device)
                b = b.permute(0,3,1,2).reshape(-1,shape[1],shape[1])
                #edit b to be shape [batch, L, L-1, dim]


                self.out_features
                ind = torch.LongTensor([[j for j in range(shape[1]) if i != j] for i in range(shape[1])]*self.out_features*shape[0]).\
                    reshape(-1,shape[1],shape[1]-1).to(self.device)

                b = torch.gather(b,2,ind).reshape(-1,self.out_features,shape[1],shape[1]-1).permute(0,2,3,1)
            else: #weight is being used, thus dim is [batch, L, L-1, dim,]
                #edit a to be shape [batch, L, L-1]
                ind = torch.LongTensor([[j for j in range(shape[1]) if i != j] for i in range(shape[1])]*shape[0]).\
                    reshape(-1,shape[1],shape[1]-1).to(self.device)
                a = torch.gather(a,2,ind).reshape(-1,shape[1],shape[1]-1)

                b = torch.einsum('ijkl, ijk -> ijkl',b,a).to(self.device)
            return b

        else:
            mask = torch.eye(shape[1], shape[1]).to(self.device)
            mask = mask.repeat(shape[0],shape[3],1,1).permute(0,2,3,1).bool()
            b = torch.einsum('ijkl, ijk -> ijkl',b,a).to(self.device)
            return b.masked_fill_(mask, 0)

class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(xW^T + b)

    Args:
        in_features (int): number of input feature :math:`x`.
        out_features (int): number of output features :math:`y`.
        bias (bool, optional): if False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): if None, no activation function is used.
        weight_init (callable, optional): weight initializer from current weight.
        bias_init (callable, optional): bias initializer from current bias.

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
        # initialize linear layer y = xW^T + b
        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y


class GetItem(nn.Module):
    """Extraction layer to get an item from SchNetPack dictionary of input tensors.

    Args:
        key (str): Property to be extracted from SchNetPack input tensors.

    """

    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: layer output.

        """
        return inputs[self.key]


class ScaleShift(nn.Module):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    """

    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y


class Standardize(nn.Module):
    r"""Standardize layer for shifting and scaling.

    .. math::
       y = \frac{x - \mu}{\sigma}

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
        eps (float, optional): small offset value to avoid zero division.

    """

    def __init__(self, mean, stddev, eps=1e-9):
        super(Standardize, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        # Add small number to catch divide by zero
        y = (input - self.mean) / (self.stddev + self.eps)
        return y


class Aggregate(nn.Module):
    """Pooling layer based on sum or average with optional masking.

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self, axis, mean=False, keepdim=True):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, input, mask=None):
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            input = input * mask[..., None]
        # compute sum of input along axis
        y = torch.sum(input, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N
        return y
