import torch
import torch.nn as nn
import numpy as np
import math

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
# input_window = 100 # number of input steps
# output_window = 1 # number of prediction steps, in this model its fixed to one
# batch_size = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)   # torch.Size([max_len, 1, d_model])
        #pe.requires_grad = False
        self.register_buffer('pe', pe) ## 매개변수로 간주하지 않기 위한 것

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1, batch_size=128, x_frames = 20, nhead=10):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.nhead = nhead
        self.feature_size = feature_size
        self.dropout = dropout
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size,5000)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_size,
                                                        nhead=self.nhead, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()
        self.batch_size = batch_size
        self.x_frames = x_frames
        self.output_linear = nn.Linear(self.x_frames, 1)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output1 = self.transformer_encoder(src,self.src_mask)#, self.src_mask)

        output2 = self.decoder(output1)

        output3 = self.output_linear(output2.view(self.batch_size, -1))

        output4 = self.sigmoid(output3.squeeze(1))

        return output4

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

