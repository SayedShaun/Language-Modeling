from dataclasses import dataclass
import math
import warnings
import torch, torch.nn.functional as F
from torch import nn, Tensor
from dataloader import Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    vocab_size: int = Tokenizer.vocab_size()
    d_model: int = 256
    seq_len: int = 512
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: int = 0.1
    head_dim: int = d_model // n_heads
    batch_size: int = 128
    flash_attention: bool = False
    

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.randn((out_features, in_features), device=device))
        self.b = nn.Parameter(torch.randn(out_features))
        # xavier init
        nn.init.xavier_uniform_(self.W)

    def forward(self, X:Tensor)->Tensor:
        if self.bias:
            return (X @ self.W.T) + self.b
        return X @ self.W.T
    

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, dropout_p:float)->None:
        super(Embedding, self).__init__()
        self.dropout_p = dropout_p
        self.W = nn.Parameter(torch.randn(vocab_size, embed_size, device=device))
    
    def forward(self, input_seq: Tensor) -> Tensor:
        embeddings = self.W[input_seq]
        return F.dropout(embeddings, p=self.dropout_p)


class LayerNorm(nn.Module):
    def __init__(self, d_model:int, epsilon:float=1e-5):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(d_model, device=device))
        self.beta = nn.Parameter(torch.zeros(d_model, device=device))
        
    def forward(self, X:Tensor)->Tensor:
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        norm_X = (X-mean) / (std + self.epsilon)
        return self.gamma * norm_X + self.beta
    

class PositionalEmbeddings(nn.Module):
    def __init__(self, config:Config):
        super(PositionalEmbeddings, self).__init__()
        self.config = config
        self.word_encoding = Embedding(config.vocab_size, config.d_model, config.dropout)
        self.position_encoding = Embedding(config.seq_len, config.d_model, config.dropout)

    def forward(self, X:Tensor)->Tensor:
        w_embedding = self.word_encoding(X)
        positions = torch.arange(X.shape[1]).unsqueeze(0).to(device)
        p_embedding = self.position_encoding(positions)
        embeddings = w_embedding + p_embedding
        return F.relu(F.dropout(embeddings, p=self.config.dropout))
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config:Config, flash_attention:bool=False):
        super(MultiHeadAttention, self).__init__()
        self.flash_attention = flash_attention
        self.config = config
        self.q_proj = Linear(self.config.d_model, self.config.d_model, bias=False)
        self.k_proj = Linear(self.config.d_model, self.config.d_model, bias=False)
        self.v_proj = Linear(self.config.d_model, self.config.d_model, bias=False)
        self.out_proj = Linear(self.config.d_model, self.config.d_model, bias=True)
        if torch.__version__ < "2.0.0":
            raise RuntimeError("Flash Attention requires PyTorch >= 2.0.0")
        if self.flash_attention==False:
            warnings.warn("Using slow attention, Use `flash_attention=True` for faster inference")

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Tensor=None)->Tensor:
        N, L, D = Q.shape
        Q, K, V = self.q_proj(Q), self.k_proj(K), self.v_proj(V)
        Q = Q.view(N, L, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        K = K.view(N, L, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        V = V.view(N, L, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        
        if self.flash_attention:
            attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        else:
            score = Q @ K.transpose(-2, -1) / math.sqrt(self.config.head_dim)
            if mask is not None:
                score = score.masked_fill(mask == 0, float(-1e6))
            weights = F.softmax(score, dim=-1)
            attention = weights @ V

        output = attention.transpose(1, 2).contiguous().view(N, L, D)
        return self.out_proj(output)
    

class MLP(nn.Module):
    def __init__(self, config:Config):
        super(MLP, self).__init__()
        self.config = config
        self.fc_1 = Linear(config.d_model, config.d_ff)
        self.fc_2 = Linear(config.d_ff, config.d_model)

    def forward(self, X:Tensor)->Tensor:
        X = F.relu(self.fc_1(X))
        X = F.dropout(X, self.config.dropout)
        return F.relu(self.fc_2(X))
    

class Block(nn.Module):
    def __init__(self, config:Config):
        super(Block, self).__init__()
        self.mha = MultiHeadAttention(config, config.flash_attention)
        self.norm_1 = LayerNorm(config.d_model)
        self.norm_2 = LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, X:Tensor, mask:Tensor=None)->Tensor:
        attention= self.mha(X, X, X, mask)
        attention = self.norm_1(attention + X)
        output = self.mlp(attention)
        return self.norm_2(output + attention)
    

if __name__ == "__main__":
    X = torch.rand(Config.batch_size, Config.seq_len, Config.d_model)
    block = Block(Config)
    print(block(X).shape)