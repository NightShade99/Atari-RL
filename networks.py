
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models 


class StateEncoder(nn.Module):
    
    def __init__(self, input_channels):
        super(StateEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)            
        self.conv4 = nn.Conv2d(64, 1024, kernel_size=3, stride=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()
        self.out_dim = 1024

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.avgpool(x)
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
        
    def forward(self, x):
        x = self.encoder(x)
        x_action, x_value = torch.split(x, self.encoder.out_dim // 2, 1)
        x_action = self.action(x_action)
        x_value = self.value(x_value)
        q_vals = x_value + (x_action - x_action.mean(-1, keepdim=True))
        return q_vals