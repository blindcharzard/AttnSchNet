import torch
from torch import nn

from schnetpack.nn import Dense,AttentionNetwork,AttentionHeads
from schnetpack.nn.base import Aggregate


__all__ = ["CFConv"]


class CFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters (int): number of filter dimensions.
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
        n_heads_weights = 0,
        n_heads_conv = 0,
        device = torch.device("cpu"),
        hyperparams = [0,0],
        dropout = 0,
        exp = False
    ):
        super(CFConv, self).__init__()
        self.device = device
        self.n_heads_weights = n_heads_weights
        self.n_heads_conv = n_heads_conv
        self.atomic_embedding_dim = n_out
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        #sum over indices
        self.agg = Aggregate(axis=axis, mean=normalize_filter)
        #added multiheaded attention to weights
        self.attention_dim = int(n_out/4) #arbitrary -> could modify at will
        if n_heads_weights > 0:
            self.Attention = AttentionHeads(n_in, self.attention_dim,n_heads=self.n_heads_weights,EXP = exp,\
                atomic_embedding_dim=n_out ,device=self.device,SM=False,hyperparams = hyperparams,dropout = dropout)
        #added multiheaded attention to convolution
        if n_heads_conv > 0:
            self.AttentionConv = AttentionHeads(n_in,self.attention_dim,n_heads=self.n_heads_conv,EXP=exp,\
                atomic_embedding_dim=n_out,device=self.device,SM=False,hyperparams = hyperparams,dropout = dropout)#for now should be single head
        #NOTE: the EXP determines if the scalar attention value should be exp(A) or just (A).
        #NOTE: exp(A) can be unstable, as can softmax below

        #add possibility to use softmax over weights
        self.softmax =  nn.Softmax(dim=3)

        #not currently used, but could add if deemed beneficial
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None,softmax = None):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1) #shape [batch, num_atoms, num_neighbors (num_atoms-1),gauusian_exp]

        #-------------NEW----------------#
        if self.n_heads_weights > 0:
            A = self.Attention(x) #attention in weight generation
            #concatenate multi-headed attention to distances
            f_ij = torch.cat((f_ij, A),dim=3)
            #f_ij = self.dropout(f_ij)
        #--------------------------------#

        # pass expanded interactomic distances through filter block
        W = self.filter_network(f_ij)
        #print(W.shape, 'Wsize')
        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer (to correct size for number of filters)
        y = self.in2f(x)
        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        #print(y.shape, 'yshape')
        # element-wise multiplication, aggregating and Dense layer
        #-----------NEW:attention in convolution--------------#
        if self.n_heads_conv > 0:
            W = self.AttentionConv(x,Weights = W) #single head to match weight size
        #added softmax
        if softmax is not None :
            W=self.softmax(W)
        #------------------------------------------------------#
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        return y

