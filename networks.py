
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models 


class StateEncoder(nn.Module):
    
    def __init__(self, projection_dim=128):
        super(StateEncoder, self).__init__()
        resnet = models.resnet18(pretrained=False)
        layers = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*layers)
        self.proj_fc = nn.Linear(512, projection_dim)
        
    def forward(self, img):
        return self.proj_fc(self.encoder(img))
    
    
class QNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x