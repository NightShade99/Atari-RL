
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models 


# class StateEncoder(nn.Module):
    
#     def __init__(self, input_channels, projection_dim=128):
#         super(StateEncoder, self).__init__()
#         resnet = models.resnet18(pretrained=False)
#         layers = list(resnet.children())[4:-1]
        
#         conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         bn1 = nn.BatchNorm2d(64)
#         relu = nn.ReLU()
#         maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
#         layers = [conv1, bn1, relu, maxpool, *layers]
        
#         self.encoder = nn.Sequential(*layers)
#         self.proj_fc = nn.Linear(512, projection_dim)
#         self.output_dim = projection_dim
        
#     def forward(self, img):
#         x = self.encoder(img)
#         x = torch.flatten(x, 1)
#         x = self.proj_fc(x)
#         return x


class StateEncoder(nn.Module):
    
    def __init__(self, input_channels, projection_dim=256):
        super(StateEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, bias=False)            
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.projection_fc = nn.Linear(4096, projection_dim)
        self.output_dim = projection_dim

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        out = self.projection_fc(x)
        return out
    
    
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