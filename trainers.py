
import os 
import gym
import utils
import wandb 
import torch
import agents
import numpy as np 
import matplotlib.pyplot as plt

from envs.envs import *
from PIL import Image
from matplotlib import animation
from envs.gym_wrappers import make_env
from gym.wrappers.monitoring.video_recorder import VideoRecorder


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
        next_state = (next_state * 255).detach().cpu().numpy().astype(np.uint8)
        done = int(done)
        return state, action, next_state, reward, done
            
    def add_sample(self, state, action, next_state, reward, done):
        state, action, next_state, reward, done = self._insert_transform(state, action, next_state, reward, done)
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state 
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
        return state, action, next_state, reward, done
    
    
class Trainer:
    
    def __init__(self, args):
        self.config, self.output_dir, self.logger, self.device = utils.initialize_experiment(
            args, output_root=f"outputs/duel_dqn/{args.env_type}/{args.env_name}"
        )
        if args.log_wandb:
            run = wandb.init(project="atari-experiments")
            self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")
            
        self.batch_size = self.config["batch_size"]
        self.best_return = -float('inf')
        self.start_epoch = 1
        
        if args.env_type == 'atari':
            self.env = AtariEnv(args.env_name, **self.config['environment'])
        elif args.env_type == 'highway':
            self.env = HighwayEnv(args.env_name, **self.config['environment'])
        elif args.env_type == 'vizdoom':
            self.env = VizdoomEnv(args.env_name, **self.config['environment'])
        
        self.agent = agents.DoubleDQN(self.config["agent"], self.env.num_actions, self.device)
        self.memory = ReplayMemory(self.config["memory_size"], self.device, self.stack_size)
        
        if args["resume"] is not None:
            self.load_state(args["resume"])
        if args["load"] is not None:
            self.load_checkpoint(args["load"])

    def save_checkpoint(self):
        torch.save(self.agent, os.path.join(self.output_dir, "checkpoint.pt"))
    
    def load_checkpoint(self, ckpt_dir):
        file = os.path.join(ckpt_dir, "checkpoint.pt")
        if os.path.exists(file):
            self.agent = torch.load(file, map_location=self.device)
            self.output_dir = ckpt_dir
            self.logger.print("Successfully loaded model checkpoint", mode="info")
        else:
            raise FileNotFoundError(f"Could not find checkpoint at {ckpt_dir}")
        
    def process_state(self, state):
        state = torch.from_numpy(state) / 255.0
        return state.to(self.device)
        
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
            utils.progress_bar(progress=(step+1)/self.config["memory_init_steps"], desc="Initializing memory", status="")
        print()
    
    def train_episode(self):
        episode_finished = False 
        total_reward = 0
        losses = []
        
        state = self.process_state(self.env.reset())
        while not episode_finished:
            action = self.agent.select_action(state)
            next_frames, reward, done, _ = self.env.step(action)
            next_state = self.process_state(next_frames)
            self.memory.add_sample(state, action, next_state, reward, done)
            total_reward += reward
            if not done:
                state = next_state
            else:
                episode_finished = True
            
            if self.agent.step % self.config["learning_interval"] == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss = self.agent.learn_from_memory(batch)
                losses.append(loss)
        
        avg_loss = 0.0 if len(losses) == 0 else sum(losses)/len(losses)
        return {"loss": avg_loss, "reward": total_reward} 
    
    @torch.no_grad()
    def eval_episode(self):
        episode_finished = False 
        total_reward = 0
        
        state = self.process_state(self.env.reset())
        while not episode_finished:
            action = self.agent.select_action(state, train=False)
            next_frames, reward, done, _ = self.env.step(action)
            next_state = self.process_state(next_frames)
            total_reward += reward
            if not done:
                state = next_state
            else:
                episode_finished = True     
        return {"reward": total_reward}
    
    @torch.no_grad()
    def create_animation(self, attempts=10):
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)

        for i in range(attempts):
            rec = VideoRecorder(self.env, path=os.path.join(self.output_dir, "videos", f"attempt_{i}.mp4"), enabled=True)
            self.agent.eval()
            total_reward = 0
            complete = False
            
            state = self.process_state(self.env.reset())
            rec.capture_frame()
            while not complete:
                action = self.agent.select_action(state, train=False)
                next_frames, reward, done, info = self.env.step(action)
                rec.capture_frame()
                next_state = self.process_state(next_frames)
                total_reward += reward
                
                if done:
                    state = self.process_state(self.env.reset())
                    if info['lives'] == 0:
                        complete = True
                else:
                    state = next_state
                    
            rec.close()
            rec.enabled = False
            self.env.close()
            print("[Attempt {:2d}] Total reward: {:.4f}".format(i, total_reward))
        
    def train(self):
        self.logger.print("Initializing memory", mode="info")
        self.initialize_memory()
        print()
        self.logger.print("Beginning training", mode="info")
        
        for epoch in range(self.start_epoch, self.config["train_epochs"]+1):
            self.agent.train()
            train_meter = utils.AverageMeter()
            desc = "[TRAIN] Epoch {:3d}/{:3d}".format(epoch, self.config["train_epochs"])
            
            for episode in range(self.config["episodes_per_epoch"]):
                train_metrics = self.train_episode()
                train_meter.add(train_metrics)
                utils.progress_bar(
                    progress=(episode+1)/self.config["episodes_per_epoch"], 
                    desc=desc, 
                    status=train_meter.return_msg()
                )

            if self.args.log_wandb:
                wandb.log({"Epoch": epoch, **train_meter.return_dict()})
            self.logger.write("Epoch {:3d}/{:3d} {}".format(
                epoch, self.config["train_epochs"], train_meter.return_msg()
            ), mode="train")
            
            if epoch % self.config["eval_every"] == 0:
                self.agent.eval()
                val_meter = utils.AverageMeter()
                desc = "{}[VALID] Epoch {:3d}/{:3d}{}".format(
                    utils.COLORS["blue"], epoch, self.config["train_epochs"], utils.COLORS["end"]
                )
                for episode in range(self.config["eval_episodes_per_epoch"]):
                    val_metrics = self.eval_episode()
                    val_meter.add(val_metrics)
                    utils.progress_bar(
                        progress=(episode+1)/self.config["eval_episodes_per_epoch"], 
                        desc=desc, 
                        status=val_meter.return_msg()
                    )

                avg_metrics = val_meter.return_dict()
                if self.args.log_wandb:
                    wandb.log({"Epoch": epoch, "Val reward": avg_metrics["reward"]})
                self.logger.record("Epoch {:3d}/{:3d} [reward] {:.4f}".format(
                    epoch, self.config["train_epochs"], avg_metrics["reward"]
                ), mode="val")
                
                if avg_metrics["reward"] > self.best_return:
                    self.best_return = avg_metrics["reward"]
                    self.save_checkpoint()                    
        print()
        self.logger.print("Completed training", mode="info")