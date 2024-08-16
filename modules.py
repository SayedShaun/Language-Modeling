from dataclasses import dataclass
import math
import warnings
import torch, torch.nn.functional as F
from torch import nn, Tensor
from dataloader import Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Require Tensorflow >= 2.16 and Keras >= 3.0.0
import tensorflow as tf
from tensorflow import keras as K


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
    

# Tensorflow Modules
class TFLinear(K.Layer):
    def __init__(self, in_size:int, out_size:int, bias:bool=True)->None:
        super(TFLinear, self).__init__()
        self.bias = bias
        self.W = K.Variable(K.random.uniform((out_size, in_size)))
        self.b = K.Variable(K.random.normal((out_size, )))

    def call(self, X:tf.Tensor)->tf.Tensor:
        if self.bias == False:
            return (X @ K.ops.transpose(self.W, (1, 0)))
        return (X @ K.ops.transpose(self.W, (1, 0)) + self.b)


class TFEmbedding(K.Layer):
    def __init__(self, vocab_size:int, embed_dim:int)->None:
        super(TFEmbedding, self).__init__()
        self.W = K.Variable(K.random.normal((vocab_size, embed_dim)))

    def call(self, X:tf.Tensor)->tf.Tensor:
        embeddings = tf.gather(self.W, X)
        return embeddings


class TFPositionalEncoding(K.Layer):
    def __init__(self, vocab_size:int, embed_dim:int, max_len:int)->None:
        super(TFPositionalEncoding, self).__init__()
        self.token_embeddings = TFEmbedding(vocab_size, embed_dim)
        self.pos_embeddings = TFEmbedding(max_len, embed_dim)

    def call(self, X:tf.Tensor)->tf.Tensor:
        t_embedding = self.token_embeddings(X)
        position = tf.range(0, X.shape[1], dtype=tf.int32)
        p_embedding = self.pos_embeddings(position)
        embeddings = t_embedding + p_embedding
        return embeddings


class TFLayerNorm(K.Layer):
    def __init__(self, d_model:int, epsilon:float=1e-5)->None:
        super(TFLayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma = K.Variable(K.ops.ones((d_model, )))
        self.beta = K.Variable(K.ops.zeros((d_model, )))

    def call(self, X:tf.Tensor)->tf.Tensor:
        mean = K.ops.mean(X, axis=-1, keepdims=True)
        var = K.ops.mean(K.ops.square(X - mean), axis=-1, keepdims=True)
        return self.gamma * (X - mean) / K.ops.sqrt(var + self.epsilon) + self.beta


class TFMultiHeadAttention(K.Layer):
    def __init__(self, config:Config)->None:
        super(TFMultiHeadAttention, self).__init__()
        self.config = Config
        self.q_proj = TFLinear(config.d_model, config.d_model)
        self.k_proj = TFLinear(config.d_model, config.d_model)
        self.v_proj = TFLinear(config.d_model, config.d_model)
        self.o_proj = TFLinear(config.d_model, config.d_model)

    def call(self, Q:tf.Tensor, K:tf.Tensor, V:tf.Tensor, mask:tf.Tensor=None)->tf.Tensor:
        B, T, C = Q.shape
        Q, K, V = self.q_proj(Q), self.k_proj(K), self.v_proj(V)
        Q = tf.reshape(Q, (B, T, self.config.n_heads, self.config.head_dim))
        K = tf.reshape(K, (B, T, self.config.n_heads, self.config.head_dim))
        V = tf.reshape(V, (B, T, self.config.n_heads, self.config.head_dim))
        Q = tf.transpose(Q, (0, 2, 1, 3))
        K = tf.transpose(K, (0, 2, 1, 3))
        V = tf.transpose(V, (0, 2, 1, 3))

        score = Q @ tf.transpose(K, (0, 1, 3, 2)) / (self.head_size ** 0.5)
        if mask is not None:
            score = K.ops.where(mask, score, -1e9)
        weights = tf.nn.softmax(score, axis=-1)
        attn_output = weights @ V
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (B, T, C))
        return self.o_proj(attn_output)


class TFMLP(K.Layer):
    def __init__(self, d_model:int, d_ff:int, dropout:float)->None:
        super(TFMLP, self).__init__()
        self.fc1 = TFLinear(d_model, d_ff)
        self.fc2 = TFLinear(d_ff, d_model)
        self.dropout = K.layers.Dropout(dropout)

    def call(self, X:tf.Tensor)->tf.Tensor:
        x = tf.nn.gelu(self.fc1(X))
        return self.fc2(self.dropout(x))


class TFEncoderBlock(K.Layer):
    def __init__(self, config: Config)->None:
        super(TFEncoderBlock, self).__init__()
        self.norm1 = TFLayerNorm(config.d_model)
        self.norm2 = TFLayerNorm(config.d_model)
        self.mha = TFMultiHeadAttention(config.d_model, config.n_heads)
        self.mlp = TFMLP(config.d_model, config.d_ff)
        self.dropout = K.layers.Dropout(config.dropout)

    def call(self, X:tf.Tensor, mask:tf.Tensor=None)->tf.Tensor:
        x = self.norm1(X)
        x = self.mha(x, x, x, mask)
        x = X + self.dropout(x)
        y = self.norm2(x)
        y = self.mlp(y)
        return x + self.dropout(y)
    

if __name__ == "__main__":
    X = torch.rand(Config.batch_size, Config.seq_len, Config.d_model)
    block = Block(Config)
    print(block(X).shape)