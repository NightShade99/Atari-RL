
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
        self.conv4 = nn.Conv2d(64, 512, kernel_size=3, stride=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()
        self.out_dim = 512

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    
class QNetwork(nn.Module):

    def __init__(self, input_channels, hidden_size, action_size):
        super(QNetwork, self).__init__()
        self.encoder = StateEncoder(input_channels)
        self.linear1 = nn.Linear(self.encoder.out_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x