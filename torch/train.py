
import os
import copy
from sympy import dotprint
import wandb
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn.functional as F

from utils import *
from networks import * 
from envs import envs
from collections import deque
from datetime import datetime as dt


class ReplayMemory:
    
    def __init__(self, capacity, num_actions, device):
        self.per_group_capacity = capacity // num_actions 
        self.mem_groups = {i: self._init_mem_group() for i in range(num_actions)}
        self.group_sizes = {i: 0 for i in range(num_actions)}
        self.device = device
    
    def _init_mem_group(self):
        return {
            'states': deque(maxlen=self.per_group_capacity),
            'actions': deque(maxlen=self.per_group_capacity),
            'rewards': deque(maxlen=self.per_group_capacity),
            'next_states': deque(maxlen=self.per_group_capacity),
            'dones': deque(maxlen=self.per_group_capacity)
        }
        
    def add(self, state, action, reward, next_state, done):
        self.mem_groups[action]['states'].append(state)
        self.mem_groups[action]['actions'].append(action)
        self.mem_groups[action]['rewards'].append(reward)
        self.mem_groups[action]['next_states'].append(next_state)
        self.mem_groups[action]['dones'].append(done)
        self.group_sizes[action] = len(self.mem_groups[action])
        
    def _sample_normal(self, batch_size):
        sizes = np.array(list(self.group_sizes.values()))
        grp_sizes = (batch_size * sizes / sizes.sum()).astype('int')
        indx = [np.random.randint(0, sizes[i], size=(grp_sizes[i],)) for i in range(len(sizes))]
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in range(len(sizes)):
            states.extend([self.mem_groups[i]['states'][j] for j in indx[i]])
            actions.extend([self.mem_groups[i]['actions'][j] for j in indx[i]])
            rewards.extend([self.mem_groups[i]['rewards'][j] for j in indx[i]])
            next_states.extend([self.mem_groups[i]['next_states'][j] for j in indx[i]])
            dones.extend([self.mem_groups[i]['dones'][j] for j in indx[i]])
            
        return (
            torch.from_numpy(np.concatenate(states)).float().to(self.device), 
            torch.from_numpy(np.array(actions)).long().to(self.device), 
            torch.from_numpy(np.array(rewards)).float().to(self.device), 
            torch.from_numpy(np.concatenate(next_states)).float().to(self.device), 
            torch.from_numpy(np.array(dones)).long().to(self.device)
        )
        
    def _sample_separated(self, batch_size):
        sizes = np.array(list(self.group_sizes.values()))
        indx = [np.random.randint(0, sizes[i], size=(batch_size,)) for i in range(len(sizes))]
        states = []

        for i in range(len(sizes)):
            states.append(
                torch.from_numpy(np.concatenate([self.mem_groups[i]['states'][j] for j in indx[i]])).float().to(self.device)
            )
        return states
    
    def sample(self, batch_size, separated=False):
        if not separated:
            return self._sample_normal(batch_size)
        else:
            return self._sample_separated(batch_size)
        
        
class Trainer:
    
    def __init__(self, args):
        self.args = args 
        self.state_shape = (args.obs_height, args.obs_width)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Environment
        if args.env_type == 'atari':
            self.env = envs.AtariEnv(
                args.env_name, args.frame_stack, True, args.clip_rewards
            )
        elif args.env_type == 'highway':
            self.env = envs.HighwayEnv(
                args.env_name, self.state_shape, args.frame_stack, args.frame_skip, args.highway_scaling
            )
        elif args.env_type == 'vizdoom':
            self.env = envs.VizdoomEnv(
                args.env_name, self.state_shape, args.frame_skip, args.frame_stack, args.vzd_screen_res, args.vzd_screen_format
            )
            
        # Action specific replay memory
        self.memory = ReplayMemory(args.mem_capacity, self.env.num_actions, self.device)
        
        # Model and optimizer
        self.encoder = StateEncoder(args.frame_stack, args.enc_feature_dim).to(self.device)
        self.online_q = QNetwork(args.enc_feature_dim, args.hidden_dim, self.env.num_actions).to(self.device)
        self.target_q = QNetwork(args.enc_feature_dim, args.hidden_dim, self.env.num_actions).to(self.device)
        self.proj_head = ProjectionHead(args.enc_feature_dim, args.proj_dim).to(self.device)
        
        params = list(self.encoder.parameters()) + list(self.q_network.parameters()) + list(self.proj_head.parameters())
        self.optim = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), amsgrad=False, weight_decay=args.weight_decay)
        
        # Initialize target Q with same params as online Q
        self.target_q.load_state_dict(self.online_q.state_dict())
        
        # Loading and experiment setup
        if args.load is not None:
            self.load(args.load)
            self.out_dir = args.load
        else:
            self.out_dir = os.path.join('outputs', args.env_type, args.env_name, dt.now().strftime('%d-%m-%Y_%H-%M'))
            self.logger = Logger(self.out_dir)
            self.logger.display_args(args)
            if args.wandb:
                run = wandb.init(project='rl-experiments-similarity')
                self.logger.write("Wandb run: {}".format(run.get_url()))
                
        # Other variables
        self.best_reward = -float('inf')
        self.epsilon = args.eps_max
        self.gamma = args.gamma
        self.tau = args.tau
        self.step = 0
                
    def save(self):
        state = {
            'encoder': self.encoder.state_dict(),
            'online_q': self.online_q.state_dict(),
            'target_q': self.target_q.state_dict(),
            'proj_head': self.proj_head.state_dict(),
            'optim': self.optim.state_dict()
        }
        torch.save(state, os.path.join(self.out_dir, 'ckpt.pth'))
        
    def load(self, ckpt_dir):
        fp = os.path.join(ckpt_dir, 'ckpt.pth')
        if os.path.exists(fp):
            state = torch.load(fp, map_location=self.device)
            self.encoder.load_state_dict(state['encoder'])
            self.online_q.load_state_dict(state['online_q'])
            self.target_q.load_state_dict(state['target_q'])
            self.proj_head.load_state_dict(state['proj_head'])
            self.optim.load_state_dict(state['optim'])
            self.logger.print(f"Loaded checkpoint from {ckpt_dir}", mode='info')
        else:
            raise FileNotFoundError(f'Could not find ckpt.pth at {ckpt_dir}')
        
    def decay_epsilon_linear(self):
        new_eps = self.args.eps_max - (self.args.eps_max - self.args.eps_min) * self.step / self.args.eps_decay_steps
        self.epsilon = max(new_eps, self.args.eps_min)
        
    @torch.no_grad()
    def select_action(self, state, stochastic=True):
        state_fs = self.encoder(state)
        
        if stochastic and random.uniform(0, 1) < self.epsilon:
            action = self.env.random_action()
            self.step += 1
            self.decay_epsilon_linear()
        else:
            action = self.online_q(state_fs).argmax(-1).item()
            
        return action 
    
    def q_learning_loss(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_state_fs = self.encoder(next_state)
            next_action = self.online_q(next_state_fs).argmax(-1)
            next_qvals = torch.gather(self.target_q(next_state_fs), -1, next_action.view(-1, 1)).squeeze()
            trg_qvals = reward + (1-done) * self.gamma * next_qvals
            
        state_fs = self.encoder(state)
        curr_qvals = torch.gather(self.online_q(state_fs), -1, action.view(-1, 1)).squeeze()
        loss = F.huber_loss(curr_qvals, trg_qvals, reduction='mean')
        return loss 
    
    def contrastive_loss(self, states, temperature):
        total_loss = 0
        state_fs, perm_state_fs = [], []
        
        # Generate permutation of the batches for positive samples
        perm_states = []
        for i in range(len(states)):
            order = torch.randperm(len(states[i])).to(self.device)
            perm_states.append(states[i][order])
            
        # Extract all the state vectors and normalize
        for i in range(len(states)):
            state_fs.append(
                F.normalize(self.proj_head(self.encoder(states[i])), p=2, dim=-1)
            )
            perm_state_fs.append(
                F.normalize(self.proj_head(self.encoder(perm_states[i])), p=2, dim=-1)
            )
            
        # Compute loss for state pairs of each action
        for i in range(len(states)):
            pos1, pos2 = state_fs[i], perm_state_fs[i]                                                          # (N_i, d), (N_i, d)
            neg1 = torch.cat([state_fs[j] for j in range(len(state_fs)) if j != i])                             # (N', d)
            neg2 = torch.cat([perm_state_fs[j] for j in range(len(perm_state_fs)) if j != i])                   # (N', d)
            neg = torch.cat([neg1, neg2], 0)                                                                    # (2N', d)
            
            pos1_pos2 = torch.diag(torch.mm(pos1, pos2.t())).view(-1, 1) / temperature                          # (N_i, 1)
            pos2_pos1 = torch.diag(torch.mm(pos2, pos1.t())).view(-1, 1) / temperature                          # (N_i, 1)
            pos1_neg = torch.mm(pos1, neg.t()) / temperature                                                    # (N_i, N')
            pos2_neg = torch.mm(pos2, neg.t()) / temperature                                                    # (N_i, N')
            
            scores_1 = torch.cat([pos1_pos2, pos1_neg], axis=1)                                                 # (N_i, 1+N')
            scores_2 = torch.cat([pos2_pos1, pos2_neg], axis=1)                                                 # (N_i, 1+N')
            scores = torch.cat([scores_1, scores_2])                                                            # (2*N_i, 1+N')
            
            logits = F.log_softmax(scores, -1)
            target = torch.zeros(logits.size(0)).to(self.device)
            loss = F.nll_loss(logits, target, reduction='mean')
            total_loss += loss
            
        return total_loss / len(states)