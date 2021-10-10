
import gym 
import torch 
import random
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms as T
from collections import deque


class AtariEnvManager:
    
    def __init__(self, device, env_name, input_size=(84, 84), stack_size=4, max_no_op=10, episodic_life=True):
        self.env = gym.make(env_name).unwrapped
        self.env.reset()
        self.env_name = env_name
        
        self.height, self.width = input_size
        self.device = device 
        self.episode_done = False
        self.life_lost = False
        self.current_screen = None 
        self.current_lives = None 
        self.noop_action = None 
        self.fire_action = None
        self.stack_size = stack_size
        self.frame_stack = deque(maxlen=stack_size)
        self.episodic_life = episodic_life
        self.max_no_op = max_no_op
        
    def reset(self):    
        self.env.reset()
        self.life_lost = False
        self.episode_done = False
        self.frame_stack = deque(maxlen=self.stack_size)
        self.init_frame_stack()
        
        # NO-OP and fire on reset
        self.noop_action = self.action_meanings.index("NOOP")
        self.fire_action = self.action_meanings.index("FIRE")
        
        # FIRE action to start off the episode
        _, _, done, init_info = self.env.step(self.fire_action)
        
        # NO-OP actions to introduce stochasticity
        for _ in range(random.randint(1, self.max_no_op)):
            _, _, done, init_info = self.env.step(self.noop_action)
            if done:
                self.env.reset()
                
        self.current_lives = init_info["ale.lives"]
    
    def just_starting(self):
        return sum(list(self.frame_stack)).sum() == 0
    
    def init_frame_stack(self):
        for _ in range(self.stack_size-1):
            self.frame_stack.append(torch.zeros(84, 84))
    
    def get_frame(self):
        return self.env.render("rgb_array")
    
    def take_action(self, action):
        if self.life_lost:
            action = self.fire_action
        _, reward, self.episode_done, info = self.env.step(action)
        if self.episodic_life:
            self.check_episodic_termination(info)
        return reward
    
    def crop_screen(self, screen):
        if "Breakout" in self.env_name or "Pong" in self.env_name:
            bbox = [34, 0, 160, 160] 
            screen = screen[:, bbox[0]:bbox[2]+bbox[0], bbox[1]:bbox[3]+bbox[1]]
        if "Gopher" in self.env_name:
            bbox = [110, 0, 120, 160]
            screen = screen[:, bbox[0]:bbox[2]+bbox[0], bbox[1]:bbox[3]+bbox[1]]
        return screen
        
    def get_state(self):
        screen = np.ascontiguousarray(self.get_frame()).transpose((2, 0, 1))
        screen = torch.from_numpy(screen).float() / 255.0
        screen = self.crop_screen(screen)
        transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.Resize((84, 84)), T.ToTensor()])
        screen = transform(screen)[0]
                
        if self.episode_done or self.life_lost:
            self.frame_stack.append(torch.zeros(84, 84))
            state = torch.stack(list(self.frame_stack), 0).unsqueeze(0).to(self.device)
            self.init_frame_stack()
        else:
            self.frame_stack.append(screen)
            state = torch.stack(list(self.frame_stack), 0).unsqueeze(0).to(self.device)
        return state
    
    def check_episodic_termination(self, info):
        if info["ale.lives"] < self.current_lives:
            self.life_lost = True 
            self.current_lives = info["ale.lives"]
        else:
            self.life_lost = False
        
    @property
    def action_size(self):
        return self.env.action_space.n 
    
    @property
    def action_meanings(self):
        return self.env.get_action_meanings()