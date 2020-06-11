import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from torchvision import datasets, models, transforms
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab, d_model)
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, dropout, n_position = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer('pos_table' self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def _get_position_angle_vec(position):
            return [position/np.power(10000, 2*( hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array(_get_position_angle_vec(position) for position in range(n_position))
        sinusoid_table[:,::2] =  np.sin(sinusoid_table[:,::2])
        sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])
        return torch.TensorFloat(sinusoid_table).unsqeeze(0)

    def forward(self ,x):
        return self.dropout(x + self.pos_table[:, :x.size(1)].clone().detach())

class FeatureExtractor(nn.Module):
    def __init__(self, d_model, submodel, name, d_features):
        super(FeatureExtractor, self).__init__()

        self.submodel = submodel
        self.name = name 
        self.fc = nn.Linear(d_features, d_model)
    
    def forward(self, x):
        for name, module in self.submodel._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.siae(1)
                return self.fc(x.view(b,c,-1).permute(0,2,1))
        return None
   
class FeedFoward(nn.Module):
    def __init__(self, d_model, d_ff, dropout =0.1):
        super(FeedFoward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.fc2(self.dropout(F.))
