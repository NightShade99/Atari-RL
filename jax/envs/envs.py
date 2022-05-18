
import os
import gym
import cv2
import highway_env
import numpy as np
import vizdoom as vzd
import itertools as it
import jax.numpy as jnp

from .gym_wrappers import * 
from collections import deque

__all__ = ['AtariEnv', 'HighwayEnv', 'VizdoomEnv']


class AtariEnv:
    
    def __init__(self, env_name, frame_stack=4, episodic_life=True, clip_rewards=False):
        self.env = gym.make(env_name)
        
        if episodic_life:
            self.env = EpisodicLifeEnv(self.env)

        self.env = NoopResetEnv(self.env, noop_max=30)
        self.env = MaxAndSkipEnv(self.env, skip=4)
        if 'FIRE' in self.env.unwrapped.get_action_meanings():
            self.env = FireResetEnv(self.env)

        self.env = WarpFrameAtari(self.env)
        if frame_stack is not None:
            self.env = FrameStack(self.env, frame_stack)
        if clip_rewards:
            self.env = ClipRewardEnv(self.env)
            
        self.num_actions = self.env.action_space.n

    def reset(self):
        state = self.env.reset()
        return jnp.expand_dims(jnp.asarray(state), 0) / 255.0
            
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = jnp.expand_dims(jnp.asarray(next_state), 0) / 255.0
        return next_state, reward, done, info
    
    def random_action(self):
        return self.env.action_space.sample()
    
    
class HighwayEnv:
    
    def __init__(self, env_name, frame_res, frame_stack, frame_skip, scaling):
        config = {
            'observation': {
                'type': 'GrayscaleObservation',
                'observation_shape': tuple(frame_res),
                'stack_size': frame_stack,
                'weights': [0.2989, 0.5870, 0.1140],
                'scaling': scaling
            },
            'policy_frequency': frame_skip
        }
        self.env = gym.make(env_name)
        self.env.configure(config)
        self.num_actions = self.env.action_space.n
        
    def reset(self):
        state = self.env.reset()
        return jnp.expand_dims(jnp.transpose(jnp.asarray(state), (1, 2, 0)), 0) / 255.0            
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = jnp.expand_dims(jnp.transpose(jnp.asarray(next_state), (1, 2, 0)), 0) / 255.0
        return next_state, reward, done, info
    
    def random_action(self):
        return self.env.action_space.sample()
    
    
class VizdoomEnv:
    
    def __init__(self, env_name, frame_res, frame_skip, frame_stack, screen_res, screen_format):
        cfg_path = os.path.join('vzd_scenarios', '{}.cfg'.format(env_name))
        if not os.path.exists(cfg_path):
            raise FileNotFoundError('Could not find vzd_scenarios/{}.cfg'.format(env_name))
        
        self.env = vzd.DoomGame()
        self.env.load_config(cfg_path)
        self.env.set_window_visible(False)
        self.env.set_mode(vzd.Mode.PLAYER)
        self.env.set_screen_format(getattr(vzd.ScreenFormat, screen_format))
        self.env.set_screen_resolution(getattr(vzd.ScreenResolution, screen_res))
        self.env.init()
        
        n = self.env.get_available_buttons_size()
        b_names = {i: str(self.env.get_available_buttons()[i]).split('.')[1] for i in range(n)}
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        
        self.action_names = {}
        for i, a in enumerate(self.actions):
            a_name = ''
            for j in range(len(a)):
                if a[j] == 1:
                    a_name += '{} '.format(b_names[j])
            if len(a_name) == 0:
                a_name = 'NO-OP'
            self.action_names[i] = a_name
            
        self.frame_res = frame_res 
        self.frame_skip = frame_skip 
        self.frame_stack = frame_stack
        self.num_actions = len(self.actions)
        self.state_deck = deque(maxlen=self.frame_stack)
        self.state_deck_unwarped = deque(maxlen=self.frame_stack)

    def _warp(self, frame):
        return cv2.resize(frame, (self.frame_res[1], self.frame_res[0]), cv2.INTER_AREA)
        
    def reset(self):
        self.env.new_episode()
        self.state_deck = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.state_deck.append(self._warp(self.env.get_state().screen_buffer))
        return jnp.expand_dims(jnp.transpose(jnp.asarray(self.state_deck), (1, 2, 0)), 0) / 255.0        
    
    def step(self, action):        
        reward = self.env.make_action(self.actions[action], self.frame_skip)
        done = self.env.is_episode_finished()
        if done:
            next_state = np.zeros_like(self.state_deck[-1])
        else:
            next_state = self._warp(self.env.get_state().screen_buffer)        
        
        self.state_deck.append(next_state)
        next_state = jnp.expand_dims(jnp.transpose(jnp.asarray(self.state_deck), (1, 2, 0)), 0) / 255.0   
        return next_state, reward, done, {}
    
    def random_action(self):
        return np.random.randint(0, self.num_actions, size=1).item()