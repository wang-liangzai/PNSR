import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import copy
import math
import torch.nn.functional as fn

from mlperf_compliance import mlperf_log

from torch.autograd import Variable
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1*torch.sin(position * div_term)
        pe[:, 1::2] = 0.1*torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = 1*Variable(self.pe[:, :x.size(1)], requires_grad=True)  # x +
        return self.dropout(x)
class Localconv(nn.Module):
    def __init__(self,embed_dim):
        super(Localconv, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=3, stride=1, padding=1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()



    def forward(self,tensor):
        conv_output = self.conv1d(tensor.permute(0,2,1)).permute(0,2,1)
        norm_output = self.layer_norm(conv_output)
        output = self.relu(norm_output)
        return output
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Multi-head attention layers
        self.W_qs = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ks = nn.Linear(input_size, hidden_size, bias=False)
        self.W_vs = nn.Linear(input_size, hidden_size, bias=False)
        self.fc = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, input):
        batch_size = input.size(0)

        # Linear projections for queries, keys, and values
        Q = self.W_qs(input)
        K = self.W_ks(input)
        V = self.W_vs(input)

        # Splitting into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)

        # Applying dropout
        attention = self.dropout(attention)

        # Weighted sum of values
        out = torch.matmul(attention, V)

        # Concatenating heads
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.hidden_size)

        # Linear projection
        out = self.fc(out)

        return out
import torch.nn.functional as F
def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 2, -1, True)
    return x * gamma / norm_tensor
class UFOAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.5):
        super(UFOAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        b_s, nq, _ = queries.shape
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        kv = torch.matmul(k, v)  # (b_s, h, nk, d_v)
        kv_norm = XNorm(kv, self.gamma)  # (b_s, h, nk, d_v)
        q_norm = XNorm(q, self.gamma)  # (b_s, h, nq, d_k)
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
class AdaptiveMixtureUnits(nn.Module):
    def __init__(self, input_size, layer_norm_eps=1e-8, hidden_dropout_prob=0.5):
        super(AdaptiveMixtureUnits, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(input_size*4,128)
        self.adaptive_act_fn = torch.sigmoid
        self.LayerNorm = nn.LayerNorm(input_size*2, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input1, input2):
        input1 = self.linear1(input1)
        input2 = self.linear1(input2)
        ada_score_alpha = self.adaptive_act_fn(input1) # [B, 1]
        ada_score_beta = 1 - ada_score_alpha
        mixture_output = torch.cat((input1 * ada_score_alpha, input2 * ada_score_beta), dim=1)  # [B, 2 * input_size]
        output = self.LayerNorm(self.dropout(self.linear2(mixture_output)))  # [B, input_size]
        return output


class Long_and_Short_term_Preference_Model(nn.Module):
    def __init__(self, nb_users, nb_items, embed_dim, mlp_layer_sizes, mlp_layer_regs):

        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError('u dummy, layer_sizes != layer_regs!')
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')

        super(Long_and_Short_term_Preference_Model, self).__init__()

        self.nb_users = nb_users
        self.nb_items = nb_items
        self.embed_dim = embed_dim
        self.nb_mlp_layers = len(mlp_layer_sizes)
        self.mlp_layer_regs = mlp_layer_regs

        self.user_embed = nn.Embedding(nb_users, embed_dim)
        self.item_embed = nn.Embedding(nb_items, embed_dim)
        self.position_embedding = nn.Embedding(2048, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim)
        self.user_embed.weight.data.normal_(0., 0.01)
        self.item_embed.weight.data.normal_(0., 0.01)
        self.localconv = Localconv(embed_dim=64)
        self.attention = UFOAttention(d_model=64, d_k=32, d_v=32, h=8)
        self.mha = MultiHeadAttention(input_size=9, hidden_size=64, num_heads=4)
        self.adaptive_fusion_module = AdaptiveMixtureUnits(input_size=64,layer_norm_eps=1e-8,hidden_dropout_prob=0.5)
        self.lstm = nn.LSTM(self.embed_dim, self.embed_dim)
        self.W_s2 = nn.Linear(embed_dim, 256, bias=True)
        self.W_s1 = nn.Linear(256, 1, bias=True)
        self.mlp0 = nn.Linear(embed_dim*2,mlp_layer_sizes[0])
        self.mlp = nn.ModuleList()
        for i in range(1, self.nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])
        self.merge = nn.Linear(mlp_layer_sizes[-1] * 2, mlp_layer_sizes[-1])
        self.final = nn.Linear(mlp_layer_sizes[-1], 1)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)
        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)

        lecunn_uniform(self.merge)
        lecunn_uniform(self.final)

    def forward(self, user, item, history,sigmoid=False):
        position_ids_item = torch.arange(item.shape[0], dtype=torch.long, device=item.device)
        position_ids_item = position_ids_item.unsqueeze(0)
        position_embedding_item = self.position_embedding(position_ids_item)
        position_embedding_item_1 = self.pos_emb(position_embedding_item)

        #************************************** global uniform module **********************************************
        xmlpu = self.user_embed(user)
        xmlpi = self.item_embed(item) +  position_embedding_item_1
        xmlpi = xmlpi.squeeze(0)
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        xmlp = self.mlp0(xmlp)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)
        xmlp = xmlp.unsqueeze(0)
        xmlp = self.attention(xmlp,xmlp,xmlp)
        xmlp = xmlp.squeeze(0)
        #******************************* local dependency enhancement module ***************************************
        xhistory = self.item_embed(history)
        x_h_i = self.item_embed(item)
        xhistory = xhistory.transpose(0,1)
        lstm_out, lstm_hidden = self.lstm(xhistory)
        lstm_out = lstm_out.transpose(0,1)
        lstm_out = self.localconv(lstm_out)
        logits = self.W_s2(lstm_out)
        logits = torch.tanh(logits)
        logits = self.W_s1(logits)
        logits = torch.transpose(logits, 1, 2)
        weights = F.softmax(logits, -1)
        atnn_out = torch.bmm(weights, lstm_out)
        size = atnn_out.size()
        atnn_out = atnn_out.view(-1, size[-1])
        rnn_out = torch.cat((atnn_out,x_h_i),dim=1)
        rnn_out = self.mlp0(rnn_out)
        for i, layer in enumerate(self.mlp):
            rnn_out = layer(rnn_out)
            rnn_out = nn.functional.relu(rnn_out)
        #*********************************** adaptive fusion module *****************************************************
        x = self.adaptive_fusion_module(rnn_out,xmlp)
        x = self.merge(x)
        x = F.relu(x)
        x = self.final(x)

        if sigmoid:
            x = torch.sigmoid(x)
        return x, weights






