import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def init_params(module, layers=1):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()

    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

    # if isinstance(module, nn.Parameter):
    #     glorot(module)
        #torch.nn.init.normal_(module, mean=0.0, std=0.02)

def xavier_uniform_3d(tensor, gain):
    size_a, size_b, size_c = tensor.size()

    for i in range(size_a):
        nn.init.xavier_uniform_(tensor[i], gain=gain)



def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerModel(nn.Module): # TransformerModel
    def __init__(self, myconfig):
        super().__init__()
        self.device = myconfig.device
        self.momentum = myconfig.momentum
        self.relu = nn.PReLU()  # nn.ReLU(inplace=True)

        self.hops = myconfig.hops  # hops= 7
        #self.pe_dim = myconfig.pe_dim
        ##1）
        self.in_Weight =  nn.Parameter(torch.ones(size=(myconfig.hops+1, myconfig.input_dim, myconfig.hidden_dim)), requires_grad=True) # <hop+1, d, d,>

        ##2）
        encoders = [
            EncoderLayer(myconfig.hidden_dim, myconfig.hidden_dim, myconfig.attention_dropout, myconfig.ffn_dropout, myconfig.n_heads)
            for _ in range(myconfig.n_layers)]  # n_layers=1
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(myconfig.hidden_dim) #!!

        ##3）
        self.attn_layer = nn.Linear(myconfig.hidden_dim*2, 1)

        ##4）
        self.scaling = nn.Parameter(torch.ones(1) * 0.5) # 未初始化！

        self.apply(lambda module: init_params(module, layers=myconfig.hops))
        xavier_uniform_3d(self.in_Weight, gain=1.414)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.momentum
            key_param.data += (1 - self.momentum) * query_param.data
        self.eval()

    def forward(self, batched_data):
        batched_data = batched_data.to(self.device)  # <B, hop, d>

        ##1）
        # adjall_embed = self.in_Linear(batched_data) # Linear -> <B, hop, d>
        node_adj_embed = batched_data.transpose(0, 1)
        node_adj_embed = torch.bmm(node_adj_embed, self.in_Weight)  # <hop, B, d>
        node_adj_embed = self.final_ln(node_adj_embed)  # !!
        node_adj_embed = node_adj_embed.transpose(0, 1)

        ##3）
        for encoderLayer in self.layers:  # EncoderLayer
            node_adj_embed = encoderLayer(node_adj_embed)  # <B, hop+1, d>

        ##2）
        # 1-hop and n-Hop
        node_tensor = node_adj_embed[:,0,:]  # <B, 1, d>
        neighbor_tensor = node_adj_embed[:, 1:, :]  # <B, hop, d>
        #neighbor_tensor = neighbor_tensor.transpose(0, 1)

        ##4） 多跳embedding 间的 attention
        node_n_embed = node_tensor.unsqueeze(dim=1).repeat(1, self.hops, 1)  # <B, hop, d>
        layer_atten = self.attn_layer(
            torch.cat((node_n_embed, neighbor_tensor), dim=2))  # <B, hop, d*2>  -> <B, hop, 1>
        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_tensor2 = neighbor_tensor * layer_atten  # <B, hop,d>
        neighbor_tensor2 = self.relu(torch.sum(neighbor_tensor2, dim=1, keepdim=True).squeeze())  # <B, hidden_dim>

        ##5）torch cat
        alpha = torch.sigmoid(self.scaling)  # Tanh
        output = node_tensor.squeeze() * alpha + neighbor_tensor2 * (1 - alpha)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, attention_dropout, ffn_dropout, n_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size) #??
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout, n_heads)
        self.self_attention_dropout = nn.Dropout(attention_dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x, attn_bias=None):
        x = self.self_attention_norm(x)  # LN(`)
        y = self.self_attention(x, x, x, attn_bias)  # MSA(`)
        y = self.self_attention_dropout(y)
        x = x + y # ??

        x = self.ffn_norm(x)
        y = self.ffn(x)
        y = self.ffn_dropout(y)
        x = x + y # ??
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.att_size = hidden_size // n_heads
        self.scale = self.att_size ** -0.5
        attout_size = n_heads * self.att_size

        self.linear_q = nn.Linear(hidden_size, attout_size)
        self.linear_k = nn.Linear(hidden_size, attout_size)
        self.linear_v = nn.Linear(hidden_size, attout_size)
        self.att_dropout = nn.Dropout(attention_dropout)

        self.output_layer = nn.Linear(attout_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.n_heads, self.att_size) # <B, hop, heads, d>
        k = self.linear_k(k).view(batch_size, -1, self.n_heads, self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.n_heads, self.att_size)

        q = q.transpose(1, 2)  # <B, heads, hop, d>
        v = v.transpose(1, 2)  # <B, heads, hop, d>
        k = k.transpose(1, 2).transpose(2, 3)  # <B, heads, d, hop>

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # <B, heads, hop, hop>
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        #x = self.att_dropout(x)
        x = x.matmul(v)  # <B, heads, hop, d>

        x = x.transpose(1, 2).contiguous()  #<B, hop, heads, d>
        x = x.view(batch_size, -1, self.n_heads * self.att_size)  #<B, hop, heads* d>

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


def drop_nei_feature(x, drop_prob):
    # if drop_prob==0:
    #     return x

    drop_mask = torch.empty(
        (x.size(1)),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


