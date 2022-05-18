
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

__all__ = ['StateEncoder', 'QNetwork', 'ProjectionHead']


class StateEncoder(nn.Module):
    
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(64, feature_dim, kernel_size=7, stride=1, bias=False)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        return x
    
    
class QNetwork(nn.Module):
    
    def __init__(self, feature_dim, hidden_dim, num_actions):
        super().__init__()
        self.q_action = nn.Sequential(
            nn.Linear(feature_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        self.q_value = nn.Sequential(
            nn.Linear(feature_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        x_value, x_action = torch.split(x, 2, dim=-1)
        x_value = self.q_value(x_value)
        x_action = self.q_action(x_action)
        qvals = x_value + (x_action - x_action.mean(-1, keepdim=True))
        return qvals 
    
    
class ProjectionHead(nn.Module):
    
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x):
        return self.head(x)