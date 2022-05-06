
import torch 
import random
import networks 
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 


class DoubleDQN:
    
    def __init__(self, config, action_size, device):
        self.step = 0
        self.config = config 
        self.action_size = action_size
        self.gamma = config.get("gamma", 0.99)
        self.eps_max, self.eps_min = self.config["eps_max"], self.config["eps_min"]
        self.eps_decay_steps = self.config["eps_decay_steps"]
        self.eps = self.eps_max
        self.update_epsilon()
        
        self.online_q = networks.QNetwork(self.config["input_channels"], self.config["hidden_size"], action_size).to(device)
        self.target_q = networks.QNetwork(self.config["input_channels"], self.config["hidden_size"], action_size).to(device)
        self.optim = optim.Adam(self.online_q.parameters(), lr=self.config["learning_rate"], betas=(0.9, 0.999), amsgrad=False)
        self.target_q.load_state_dict(self.online_q.state_dict())
        
    def train(self):
        self.online_q.train()
        self.target_q.eval()
        
    def eval(self):
        self.online_q.eval()
        self.target_q.eval()
        
    def update_epsilon(self):
        if self.step <= self.eps_decay_steps:
            self.eps = self.eps_max - (self.step / self.eps_decay_steps) * (self.eps_max - self.eps_min) 
        else:
            self.eps = self.eps_min
            
    def update_target_model(self):
        if self.step % self.config["target_update_interval"] == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())
    
    @torch.no_grad()
    def select_action(self, state, train=True):
        self.step = self.step+1 if train else self.step
        if random.uniform(0, 1) < self.eps:
            action = random.choice(np.arange(self.action_size))
        else:
            action = self.online_q(state).argmax(-1).item()
        self.update_epsilon()
        self.update_target_model()
        return action
    
    def learn_from_memory(self, batch):
        state, action, next_state, reward, done = batch 
        pred_q = self.online_q(state).gather(1, action.view(-1, 1)).squeeze(-1)
        
        with torch.no_grad():
            next_action = self.online_q(next_state).argmax(-1)
            next_values = self.target_q(next_state).gather(1, next_action.view(-1, 1)).squeeze(-1)
            trg_q = torch.sign(reward) + (1 - done) * self.gamma * next_values

        self.optim.zero_grad()
        loss = F.huber_loss(pred_q, trg_q).float()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_q.parameters(), 1.0)
        self.optim.step()
        return loss.item()