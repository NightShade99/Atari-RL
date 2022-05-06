
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import jax
import jax.numpy as jnp 
import flax.linen as fnn 

__all__ = [
    'StateEncoder', 'QNetwork', 'MultiHeadSelfAttention', 'ViTLayer', 'ViTModel'
]


class StateEncoder(nn.Module):
    
    def __init__(self, input_channels):
        super(StateEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)            
        self.conv4 = nn.Conv2d(64, 1024, kernel_size=7, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.out_dim = 1024

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        return x
    
    
class QNetwork(nn.Module):
    """ Dueling DQN architecture """

    def __init__(self, input_channels, hidden_dim, action_size):
        super(QNetwork, self).__init__()
        self.encoder = StateEncoder(input_channels)
        self.value = nn.Sequential(
            nn.Linear(self.encoder.out_dim // 2, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
        self.action = nn.Sequential(
            nn.Linear(self.encoder.out_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )
        self.init_network()
        
    def init_network(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        x_action, x_value = torch.split(x, self.encoder.out_dim // 2, 1)
        x_action = self.action(x_action)
        x_value = self.value(x_value)
        q_vals = x_value + (x_action - x_action.mean(-1, keepdim=True))
        return q_vals
    
    
def unfold_img_to_sequence(inp, patch_size):
    assert inp.shape[1] % patch_size == 0, f'Input height {inp.shape[1]} not divisible by {patch_size}'
    assert inp.shape[2] % patch_size == 0, f'Input width {inp.shape[2]} not divisible by {patch_size}'
    
    sequence = []
    bs, h, w, _ = inp.shape
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            p = inp[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :].reshape(bs, -1)
            sequence.append(p)            
    
    return jnp.stack(sequence, 1)


class MultiHeadSelfAttention(fnn.Module):
    num_heads: int = 1
    model_dim: int = 512
    dropout_rate: float = 0.1
    
    def setup(self):
        kernel_init = fnn.initializers.xavier_normal() 
        
        self.layernorm = fnn.LayerNorm()
        self.dropout = fnn.Dropout(self.dropout_rate)
        self.query = fnn.Dense(self.model_dim, use_bias=False, kernel_init=kernel_init)
        self.key = fnn.Dense(self.model_dim, use_bias=False, kernel_init=kernel_init)
        self.value = fnn.Dense(self.model_dim, use_bias=False, kernel_init=kernel_init)
        
    def __call__(self, x, training=True):
        bs, sl, _ = x.shape 
        head_dim = self.model_dim // self.num_heads
        
        x_norm = self.layernorm(x)
        q = self.query(x_norm).reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        k = self.key(x_norm).reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        v = self.value(x_norm).reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        
        attn_scores = jnp.einsum('bhid,bhjd->bhij', q, k)
        attn_probs = fnn.softmax(attn_scores / jnp.sqrt(head_dim), axis=-1)
        attn_probs = self.dropout(attn_probs, deterministic=(not training))
        
        out = jnp.einsum('bhij,bhjd->bhid', attn_probs, v)
        out = out.transpose((0, 2, 1, 3)).reshape(bs, sl, self.model_dim)
        return out + x, attn_probs
    
    
class ViTLayer(fnn.Module):
    num_heads: int = 1
    model_dim: int = 512
    mlp_hidden_dim: int = 2048 
    attn_dropout_rate: float = 0.1
    
    def setup(self):
        kernel_init = fnn.initializers.xavier_normal()
        bias_init = fnn.initializers.normal(stddev=1e-06)
        
        self.attention = MultiHeadSelfAttention(self.num_heads, self.model_dim, self.attn_dropout_rate)
        self.mlp_fc1 = fnn.Dense(self.mlp_hidden_dim, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_fc2 = fnn.Dense(self.model_dim, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_layernorm = fnn.LayerNorm()
        
    def __call__(self, x, training=True):
        x, attn_probs = self.attention(x, training)
        x_norm = self.mlp_layernorm(x)
        x_norm = self.mlp_fc2(fnn.gelu(self.mlp_fc1(x_norm)))
        return x_norm + x, attn_probs
    
    
class ViTModel(fnn.Module):
    num_actions: int
    num_heads: int = 1
    num_layers: int = 4
    patch_size: int = 4
    model_dim: int = 512
    mlp_hidden_dim: int = 2048
    attn_dropout_rate: float = 0.1
    
    @fnn.compact
    def __call__(self, inp, training=True):
        # Convert image to sequence of flattened patches
        x = unfold_img_to_sequence(inp, self.patch_size)
        bs, seqlen, inp_fdim = x.shape 
         
        # Append CLS token and scale features up to model_dim
        cls_token = self.param('cls', fnn.initializers.zeros, (bs, 1, inp_fdim))
        x = jnp.concatenate([cls_token, x], axis=1)
        x = fnn.Dense(self.model_dim)(x)
        
        # Add positional embeddings
        pos_embeds = fnn.Embed(seqlen+1, self.model_dim)(jnp.arange(seqlen+1))
        x = x + pos_embeds 
        
        # Pass through self attention layers 
        layerwise_attn_probs = {}
        for i in range(self.num_layers):
            x, attn_probs = ViTLayer(self.num_heads, self.model_dim, self.mlp_hidden_dim, self.attn_dropout_rate)(x, training)
            layerwise_attn_probs[i] = attn_probs
            
        # Classifier
        preds = fnn.Dense(self.num_actions)(x[:, 0, :])
        
        return preds, layerwise_attn_probs