
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

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
    
    
class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, num_heads, model_dim, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // self.num_heads
        
        self.layernorm = nn.LayerNorm(model_dim)
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate) 
        
    def forward(self, x):
        bs, seqlen, _ = x.size()
        x_norm = self.layernorm(x)
        
        q = self.query(x_norm).view(bs, seqlen, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = self.key(x_norm).view(bs, seqlen, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = self.value(x_norm).view(bs, seqlen, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        
        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k)
        attn_probs = F.softmax(attn_scores / math.sqrt(self.head_dim), -1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.einsum('bhij,bhjd->bhid', attn_probs, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(bs, seqlen, -1) + x
        return out, attn_probs
    
    
class ViTLayer(nn.Module):
    
    def __init__(self, num_heads, model_dim, mlp_hidden_dim, attn_dropout_rate):
        super().__init__()
        self.attention = MultiHeadSelfAttention(num_heads, model_dim, attn_dropout_rate)
        self.mlp_fc1 = nn.Linear(model_dim, mlp_hidden_dim)
        self.mlp_fc2 = nn.Linear(mlp_hidden_dim, model_dim)
        self.layernorm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        x, attn_probs = self.attention(x)
        x_norm = self.layernorm(x)
        out = self.mlp_fc2(F.gelu(self.mlp_fc1(x_norm))) + x
        return out, attn_probs
    
    
class ViTModel(nn.Module):
    
    def __init__(
        self, num_layers, num_heads, num_actions, patch_size, in_channels, 
        seqlen, model_dim, mlp_hidden_dim, attn_dropout_rate
    ):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.upscale = nn.Linear(in_channels * patch_size ** 2, model_dim)
        self.embedding = nn.Embedding(seqlen+1, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim), requires_grad=True)
        
        self.vit_layers = nn.ModuleList([
            ViTLayer(num_heads, model_dim, mlp_hidden_dim, attn_dropout_rate) for _ in range(num_layers)
        ])
        self.cls_head = nn.Linear(model_dim, num_actions)
        self.num_layers = num_layers
        
    def forward(self, obs):
        x = self.unfold(obs).transpose(1, 2).contiguous()
        x = self.upscale(x)
        
        idx = torch.arange(x.size(1)+1).to(x.device)
        x = torch.cat([self.cls_token, x], 1) + self.embedding(idx)
        
        layerwise_attn = {}
        for i in range(self.num_layers):
            x, attn_probs = self.vit_layers[i](x)
            layerwise_attn[i] = attn_probs
            
        out = self.cls_head(x[:, 0, :])
        return out, layerwise_attn