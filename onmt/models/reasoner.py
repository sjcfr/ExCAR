"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hgcn.manifolds as manifolds
from hgcn.manifolds.poincare import PoincareBall
from hgcn.layers.att_layers import GraphAttentionLayer
import hgcn.layers.hyp_layers as hyp_layers
from hgcn.layers.layers import GraphConvolution, Linear, get_dim_act
import hgcn.utils.math_utils as pmath

import copy

import pdb

class PotentialFunction(nn.Module):
    def __init__(self, args):
        super(PotentialFunction, self).__init__()
        
        if args.reasoner == 'cmnln':
            input_dim  = args.feat_dim * 2
        else:
            input_dim  = args.feat_dim
            
        output_dim  = args.feat_dim
        
        self.state_updater_c = nn.Linear(input_dim, output_dim)
        self.state_updater_e = nn.Linear(input_dim, output_dim)
        
        self.cs_calc = nn.Linear(output_dim, output_dim, bias=False)
    
    def forward(self, h_c, h_e):
        
        h_c = self.state_updater_c(h_c)
        h_e = self.state_updater_c(h_e)
        
        cs = torch.sigmoid((self.cs_calc(h_c) * h_e).sum(-1)).unsqueeze(-1)
        
        return cs


class Reasoner(nn.Module):
    """
    CMNLN/MLN/fixed causal strength reasoner
    """

    def __init__(self, args):
        super(Reasoner, self).__init__()
        
        self.args = args
        
        self.potential_function = PotentialFunction(args)
        
        if self.args.reasoner == 'cmnln':
            self.anti_updater = nn.Linear(args.feat_dim * 2, args.feat_dim, bias=False)
        
        self.output_layer = nn.Linear(args.feat_dim * 2, 2) 
        
    def forward(self, x, end_inds):

        max_chain = self.args.max_chain
        evi_l = self.args.max_evi_length

        s_anti = x[:,0:1,:]
        
        cs_ls = []
        s_anti_ls = []
        
        for i in range(x.shape[1]-1):
        
            h_c_0 = x[:,i:(i+1),]
            h_e_0 = x[:,(i+1):(i+2),]
        
            if self.args.reasoner == 'cmnln':
                h_c = torch.cat([s_anti, h_c_0], -1)
                h_e = torch.cat([s_anti, h_e_0], -1)
                
                cs = self.potential_function(h_c, h_e)
                cs_ls.append(cs)
                
                s_anti = torch.tanh(cs * self.anti_updater(torch.cat([h_c_0, h_e_0], -1))) + s_anti
                s_anti_ls.append(s_anti)
            
            elif self.args.reasoner in ['mln', 'fixed']:
                h_c = x[:,i:(i+1),:]
                h_e = x[:,(i+1):(i+2),:]
                
                cs = self.potential_function(h_c, h_e)                
                cs_ls.append(cs)
                
                s_anti = torch.tanh(cs * (h_c_0 + h_e_0)) + s_anti
                s_anti_ls.append(s_anti)
        
        cs_ls = torch.cat(cs_ls, -1)
        s_anti_ls = torch.cat(s_anti_ls, 1)
        
        cum_cs = torch.cumprod(cs_ls, -1)
        
        if self.args.reasoner == 'fixed':
            cum_cs = cum_cs * 0 + 0.5
        
        cum_cs = cum_cs.squeeze()    
        # each chain contains #max_chain_evi evidences, so that it contains #max_chain_evi-1 rules
        end_inds = end_inds[:,1:]
        
        # (cum_cs * end_inds).sum(-1): have a length of #batch_size * #max_chain
        cum_cs = (cum_cs * end_inds).sum(-1) - (1 - end_inds.sum(-1))  * 10000 
        # -(1 - end_inds.sum(-1)) * 10000: for the padded evidence chains, their causal strength should equals to 0. hence, 
        # before the softmax operation, for the representation of padded evidence chains, we  
        cum_cs = cum_cs.reshape(-1, max_chain)
        cum_cs = torch.softmax(cum_cs, -1)
        
        end_inds = end_inds.unsqueeze(-1).expand(s_anti_ls.shape)
        s_anti_ls = (s_anti_ls * end_inds).sum(1)
        
        cum_cs = cum_cs.unsqueeze(-1)
        s_anti_ls = s_anti_ls.reshape(int(s_anti_ls.shape[0] / max_chain), max_chain, -1).transpose(2, 1)
        
        u = torch.bmm(s_anti_ls, cum_cs).squeeze()

        u = torch.cat([x[::self.args.max_chain,0], u], dim=1)
        
        scores = self.output_layer(u)
        
        return scores
        
        
        
        
                
                
        

        
        

