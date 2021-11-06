"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/users4/ldu/causal_inference/')
from onmt.BertModules import *
from onmt.models.reasoner import *

import pdb

class XCARModel(nn.Module):
    """
    Graph based calsual inference model.
    """

    def __init__(self, args, lm_encoder):
        super(XCARModel, self).__init__()
        self.reasoner = Reasoner(args) # its correct.
        self.lm_encoder = lm_encoder
        self.args = args
        
    def forward(self, input_ids, end_inds, sentence_mask=None, attention_mask=None, token_type_ids=None, disturb=False):
        #pdb.set_trace()
        
        batch_size = input_ids.shape[0]
        max_chain = self.args.max_chain
        evi_l = self.args.max_evi_length
        
        input_ids = input_ids.reshape(batch_size * max_chain, -1)
        attention_mask = attention_mask.reshape(batch_size * max_chain, -1)
        token_type_ids = token_type_ids.reshape(batch_size * max_chain, -1)
        sentence_mask = sentence_mask.reshape(batch_size * max_chain, -1)
        end_inds = end_inds.reshape(batch_size * max_chain, -1)
        
        event_embeddings = self.lm_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, disturb=disturb)[0][-1] 
        
        sentence_mask = sentence_mask.unsqueeze(-1)
        sentence_mask = sentence_mask.expand(sentence_mask.shape[0], sentence_mask.shape[1], event_embeddings.shape[-1])
        
        #end_inds = end_inds.unsqueeze(-1)
        #end_inds = end_inds.expand(end_inds.shape[0], end_inds.shape[1], event_embeddings.shape[-1])
        
        event_embeddings = torch.tanh(event_embeddings * sentence_mask) 
        
        event_embeddings = event_embeddings[:,::evi_l, :]
        
        scores = self.reasoner(event_embeddings, end_inds)
                
        return scores
        
