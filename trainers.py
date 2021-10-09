
import os 
import gym
import utils
import wandb 
import torch
import agents 
import numpy as np 
from PIL import Image
from gym_wrappers import make_env


class ReplayMemory:
    
    def __init__(self, memory_size, device, stack_size=4):
        self.ptr = 0
        self.filled = 0
        self.device = device
        self.stack_size = stack_size
        self.mem_size = memory_size
        
        self.states = np.zeros((self.mem_size, self.stack_size, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((self.mem_size, self.stack_size, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((self.mem_size,), dtype=np.uint8)
        self.rewards = np.zeros((self.mem_size,), dtype=np.float32)
        self.terminal = np.zeros((self.mem_size,), dtype=np.uint8) 
            
    def _insert_transform(self, state, action, next_state, reward, done):
        state = (state * 255).detach().cpu().numpy().astype(np.uint8)
        next_state = (next_state * 255).detach().cpu().numpy(astype.uint8)
        done = int(done)
        return state, action, next_state, reward, done
            
    def add_sample(self, state, action, next_state, reward, done):
        state, action, next_state, reward, done = self._insert_transform(state, action, next_state, reward, done)
        self.states[self.ptr] = state
        self.next_state[self.ptr] = next_state 
        self.actions[self.ptr] = action 
        self.rewards[self.ptr] = reward 
        self.terminal[self.ptr] = done 
        self.ptr = (self.ptr + 1) % self.mem_size
        self.filled = min(self.filled+1, self.mem_size)
        
    def get_batch(self, batch_size):
        assert batch_size < self.filled, "Not enough samples yet"
        idx = np.random.choice(np.arange(self.filled), size=batch_size, replace=False)
        
        state = torch.from_numpy(self.states[idx] / 255.0).float().to(self.device)         
        next_state = torch.from_numpy(self.next_states[idx] / 255.0).float().to(self.device)         
        action = torch.from_numpy(self.actions[idx]).long().to(self.device)         
        reward = torch.from_numpy(self.rewards[idx]).float().to(self.device)         
        done = torch.from_numpy(self.terminal[idx]).float().to(self.device)         
        return obs, action, next_obs, reward, done
    
    
class Trainer:
    
    def __init__(self, args):
        self.config, self.output_dir, self.logger, self.device = utils.initialize_experiment(args, output_root="outputs/double_q/breakout")
        run = wandb.init(project="atari-experiments")
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")
        self.stack_size = self.config["frames_per_sample"]
        self.batch_size = self.config["batch_size"]
        self.best_return = 0
        
        self.env = make_env(env=gym.make(self.config["environment"].pop("env_name")), **self.config["environment"])
        self.agent = agents.DoubleDQN(self.config["agent"], self.env.action_space.n, self.device)
        self.memory = ReplayMemory(self.config["memory_size"], self.device, self.stack_size)
                
    def save_checkpoint(self):
        torch.save(self.agent, os.path.join(self.output_dir, "checkpoint.pt"))
    
    def load_checkpoint(self, ckpt_dir):
        file = os.path.join(ckpt_dir, "checkpoint.pt")
        if os.path.exists(file):
            self.agent = torch.load(file, map_location=self.device)
            self.logger.print("Successfully loaded model checkpoint", mode="info")
        else:
            raise FileNotFoundError(f"Could not find checkpoint at {ckpt_dir}")
        
    def process_state(self, frames):
        state = np.array(frames).transpose((2, 0, 1))
        state = torch.from_numpy(state) / 255.0
        return state.unsqueeze(0).to(self.device)
        
    def initialize_memory(self):
        state = self.process_state(self.env.reset())
        
        for step in range(self.config["memory_init_steps"]):
            action = self.env.action_space.sample()
            next_frames, reward, done, _ = self.env.step(action)
            next_state = self.process_state(next_frames)
            self.memory.add_sample(state, action, next_state, reward, done)
            state = next_state if not done else self.process_state(self.env.reset())
            
            if step % self.config["learning_interval"] == 0 and (self.memory.filled > self.batch_size):
                batch = self.memory.get_batch(self.batch_size)
                _ = self.agent.learn_from_memory(batch)
                self.agent.learning_steps += 1              
            utils.progress_bar(progress=(step+1)/self.config["memory_init_steps"], desc="Initializing memory", status="")       
        print()
    
    def train_episode(self):
        episode_finished = False 
        total_reward = 0
        losses = []
        
        state = self.process_state(self.env.reset())
        while not episode_finished:
            action = self.model.select_action(state)
            next_frames, reward, done, _ = self.env.step(action)
            next_obs = self.process_state(next_frames)
            self.memory.add_sample(to_numpy(obs), to_numpy(action), to_numpy(next_obs), to_numpy(reward), to_numpy(done))
            total_reward += reward
            if not done:
                state = next_state
            else:
                episode_finished = True
            
            if self.agent.action_steps % self.config["learning_interval"] == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss = self.model.learning_step(batch)
                self.agent.learning_steps += 1
                losses.append(loss) 
                
        return {"loss": sum(losses)/len(losses), "reward": total_reward} 
    
    @torch.no_grad()
    def eval_episode(self):
        episode_finished = False 
        total_reward = 0
        
        state = self.process_state(self.env.reset())
        while not episode_finished:
            action = self.model.select_action(state, train=False)
            next_frames, reward, done, _ = self.env.step(action)
            next_obs = self.process_state(next_frames)
            total_reward += reward
            if not done:
                state = next_state
            else:
                episode_finished = True            
        return total_reward
    
    def train(self):
        self.logger.print("Initializing memory", mode="info")
        self.initialize_memory()
        print()
        self.logger.print("Beginning training", mode="info")
        
        for episode in range(1, self.config["train_episodes"]+1):    
            self.model.train()
            train_metrics = self.train_episode()
            wandb.log({f"Train {key}": value for key, value in train_metrics.items()})
            if episode % self.config["logging_interval"] == 0:
                self.logger.record("Episode {:5d}/{:5d} {}".format(
                    episode, self.config["train_episodes"], " ".join(["[{}] {:.4f}".format(k, v) for k, v in train_metrics.items()])), mode="train")

            if episode % self.config["eval_every"] == 0:
                val_rewards = []
                for i in range(self.config["eval_episodes"]):
                    reward = self.eval_episode()
                    val_rewards.append(reward)
                avg_reward = sum(val_rewards)/len(val_rewards)
                wandb.log({"Val reward": avg_reward, "Episode": episode})
                self.logger.record("Episode {:5d}/{:5d} [Reward] {:.4f}".format(episode, self.config["train_episodes"], avg_reward), mode="val")
                
                if avg_reward > self.best_return:
                    self.best_return = avg_reward
                    self.save_checkpoint()                    
        print()
        self.logger.print("Completed training", mode="info")