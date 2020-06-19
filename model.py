import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from torchvision import datasets, models, transforms

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
                c = x.size(1)
                return self.fc(x.view(b,c,-1).permute(0,2,1))
        return None
   
class PositionwiseFeedFoward(nn.Module):
    def __init__(self, d_model, d_ff, dropout =0.1):
        super(FeedFoward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(0)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model),4)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        #do all the linear projections to d_model => h x d_k
        query, key, value = [l(x).view(n_batches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query, key, value))]
        #scaled dot-product attention
        x, self.attn = attention(query, key, value, mask = mask, dropout=self.dropout)
         #concat h1,h2...h8
        x = x.transpose(1,2).contiguous().view(n_batches,-1,self.h*self.d_k)

        return self.linears[-1](x)
class layerNorm(nn.Module):
    def __init__(self, features, eps =1e-6):
        super(layerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) +self.b_2
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = layerNorm(size)
        self.dropout = dropout
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.feedforward = feedforward 
        self.layer = clones(SublayerConnection(size,dropout),2)
    def forward(self, x, mask):
        return self.layerNorm[1](self.layer[0](x, lambda x:self_attn(x,x,x,mask)), self.feedforward)
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = layerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.layers = clones(SublayerConnection(size, dropout), 3)
        self.feedforward = feedforward
    def feedforward(self, x, memory, src_mask, tgt_mask):
        m =memory
        x = self.layers[0](x, lambda x: self_attn(x, x, x, src_mask))
        x = self.layers[1](x, lambda x,m: src_attn(x, m, m, tgt_mask))
        return self.layers[2](x, self.feedforward)
class Decoder(nn.Module):
    def __init__(self, layer, mask, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = layerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.fc = torch.Linear(d_model, vocab)
    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_emb, tgt_emb, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.generator = generator
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, tgt, tgt_mask )
    def encode(self, src, src_mask)
        return self.encode(src,src_mask)
    def decode(self, src, src_mask, tgt, tgt_mask):
        return self.decode(tgt, src, src_mask, tgt_emb)
def _make_model(tgt_vocab, d_model=256, d_ff = 1024, d_features =1024, h=8, dropout=0.1, N=4)
    resnet = models.resnet101(pretrained= False)

    c = copy.deepcopy

    ff = PositionwiseFeedFoward(d_model, d_ff, dropout)

    attn = MultiHeadedAttention(d_model, h, dropout)

    pe = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        nn.Sequential(FeatureExtractor(d_model, resnet, 'layer3', d_features), c(pe))
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(pe))
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model



