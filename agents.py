
import torch 
import random
import networks 
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 


class DoubleDQN:
    
    def __init__(self, config, action_space_size, device):
        self.config = config 
        self.action_size = action_space_size
        self.gamma = config.get("gamma", 0.9)
        self.eps, self.eps_min = config.get("epsilon", 1.0), config.get("epsilon_min", 1e-05)
        self.eps_decay_rate = config.get("epsilon_decay", 0.9)
        
        self.encoder = networks.StateEncoder(config["input_channels"], config["projection_dim"]).to(device)
        self.online_q = networks.QNetwork(self.encoder.output_dim, config["hidden_size"], action_space_size).to(device)
        self.target_q = networks.QNetwork(self.encoder.output_dim, config["hidden_size"], action_space_size).to(device)
        self.optim = optim.Adam(list(self.encoder.parameters())+list(self.online_q.parameters()), **config["optim"])
        for p in self.target_q.parameters():
            p.requires_grad = False
        
    def train(self):
        self.encoder.train()
        self.online_q.train()
        
    def eval(self):
        self.encoder.eval()
        self.online_q.eval()
        
    def select_action(self, obs):
        state = self.encoder(obs)
        q_vals = self.online_q(state)
        if random.uniform(0, 1) < self.eps:
            action = random.choice(np.arange(self.action_size))
        else:
            action = q_vals.argmax(-1).item()
        return action
    
    def decay_epsilon(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay_rate
        else:
            self.eps = self.eps_min
            
    def update_target_critic(self):
        self.target_q.load_state_dict(self.online_q.state_dict())
    
    def learning_step(self, step, batch):
        obs, action, next_obs, reward, done = batch 
        state, next_state = self.encoder(obs), self.encoder(next_obs)
        
        pred_q = self.online_q(state).gather(1, action.view(-1, 1)).squeeze(-1)
        with torch.no_grad():
            next_action = self.online_q(next_state).argmax(-1)
            next_q_trg = self.target_q(next_state).gather(1, next_action.view(-1, 1)).squeeze(-1)
            trg_q = reward.view(-1,) + (1 - done.view(-1,)) * self.gamma * next_q_trg

        self.optim.zero_grad()
        loss = F.mse_loss(pred_q, trg_q).float()
        loss.backward()
        self.optim.step()
        return {"loss": np.log10(loss.item())}    