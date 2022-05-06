
import os 
import cv2
import time
import utils
import wandb 
import torch
import agents
import pickle
import warnings
import numpy as np 
import vizdoom as vzd
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from networks import *
from envs.envs import *
from gym.wrappers.monitoring.video_recorder import VideoRecorder

warnings.filterwarnings(action='ignore')


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
            args, output_root=f"outputs/duel_dqn/{args.env_type}/{args.env_name}", ckpt_dir=args.load
        )
        if args.log_wandb:
            run = wandb.init(project="atari-experiments")
            self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")
            
        self.batch_size = self.config["batch_size"]
        self.best_return = -float('inf')
        self.start_epoch = 1
        self.args = args
        
        if args.env_type == 'atari':
            self.env = AtariEnv(args.env_name, **self.config['environment'])
        elif args.env_type == 'highway':
            self.env = HighwayEnv(args.env_name, **self.config['environment'])
        elif args.env_type == 'vizdoom':
            self.env = VizdoomEnv(args.env_name, **self.config['environment'])
        
        self.agent = agents.DoubleDQN(self.config["agent"], self.env.num_actions, self.device)
        self.memory = ReplayMemory(self.args.memory_size, self.device, self.config['environment']['frame_stack'])
        
        if args.resume is not None:
            self.load_state(args.resume)
        if args.load is not None:
            self.load_checkpoint(args.load)

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
        state = torch.from_numpy(state / 255.0).float()
        return state.to(self.device)
        
    def initialize_memory(self):
        state = self.process_state(self.env.reset())
        
        for step in range(self.config["memory_init_steps"]):
            action = self.env.random_action()
            next_frames, reward, done, _ = self.env.step(action)
            next_state = self.process_state(next_frames)
            self.memory.add_sample(state, action, next_state, reward, done)
            state = next_state if not done else self.process_state(self.env.reset())
            
            if step % self.config["learning_interval"] == 0 and (self.memory.filled > self.batch_size):
                batch = self.memory.get_batch(self.batch_size)
                _ = self.agent.learn_from_memory(batch)
            utils.progress_bar(progress=(step+1)/self.config["memory_init_steps"], desc="Initializing memory", status="")
    
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
    def collect_experience(self):
        states, actions = [], []
        state = self.env.reset()
        
        for _ in tqdm(range(self.args.num_samples)):
            action = self.agent.select_action(self.process_state(state), train=False)
            states.append(state), actions.append(action)
            next_state, _, done, _ = self.env_step(action)
            state = next_state if not done else self.env.reset()
            
        savedata = {'states': np.concatenate(states, 0), 'actions': np.concatenate(actions, 0)}
        with open(os.path.join(self.args.dset_save_dir, self.args.env_name), 'wb') as f:
            pickle.dump(savedata, f)
    
    @torch.no_grad()
    def create_animation(self, attempts=10):
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)

        for i in range(attempts):
            rec = VideoRecorder(self.env.env, path=os.path.join(self.output_dir, "videos", f"attempt_{i}.mp4"), enabled=True)
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
                    complete = True
                else:
                    state = next_state
                    
            rec.close()
            rec.enabled = False
            self.env.env.close()
            print("[Attempt {:2d}] Total reward: {:.4f}".format(i, total_reward))
            
    @torch.no_grad()
    def create_vzd_animation(self):
        self.env.env.close()
        self.env.env.set_window_visible(True)
        self.env.env.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.env.env.init()
        self.agent.eval()
        os.makedirs(os.path.join(self.output_dir, 'videos'), exist_ok=True)
        
        for i in range(5):
            state = self.process_state(self.env.reset())
            
            while True:
                action = self.agent.select_action(state, train=False)
                next_state, _, done, _ = self.env.step(action, train=False)
                if not done:
                    state = self.process_state(next_state)
                else:
                    break
                    
            time.sleep(1.0)
            total_reward = self.env.env.get_total_reward()
            self.logger.print('Episode: {:3d} - Reward: {:4d}'.format(i+1, round(total_reward)))
            
        self.env.env.close()
        
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
    
    
class AttentionTrainer:
    
    def __init__(self, args):
        self.config, self.output_dir, self.logger, self.device = utils.initialize_experiment(
            args, output_root=f"outputs/attention/{args.env_type}/{args.env_name}", ckpt_dir=args.load
        )
        if args.log_wandb:
            run = wandb.init(project="atari-experiments")
            self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")
            
        self.start_epoch = 1
        self.best_acc = 0
        self.args = args
        
        if args.env_type == 'atari':
            self.env = AtariEnv(args.env_name, **self.config['environment'])
        elif args.env_type == 'highway':
            self.env = HighwayEnv(args.env_name, **self.config['environment'])
        elif args.env_type == 'vizdoom':
            self.env = VizdoomEnv(args.env_name, **self.config['environment'])
            
        obs_shape = (4, 84, 84)
        self.h_patches, self.w_patches = (obs_shape[1] // args.patch_size), (obs_shape[2] // args.patch_size)   
        num_patches = self.h_patches * self.w_patches
            
        # Attention model
        self.model = ViTModel(
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_actions=self.env.num_actions,
            patch_size=args.patch_size,
            in_channels=obs_shape[0],
            seqlen=num_patches,
            model_dim=args.model_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            attn_dropout_rate=args.attn_dropout_rate
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs, eta_min=1e-10)
        
        assert args.load is not None, f'Agent checkpoint needed for attention training'            
        self.agent = torch.load(os.path.join(args.load, 'checkpoint.pt'), map_location=self.device)
        self.memory = ReplayMemory(self.config["memory_size"], self.device, self.config['environment']['frame_stack'])

    def save_checkpoint(self):
        state = {
            'model': self.model.state_dict(), 
            'optim': self.optimizer.state_dict(), 
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.output_dir, "checkpoint.pt"))
    
    def load_checkpoint(self, ckpt_dir):
        file = os.path.join(ckpt_dir, "checkpoint.pt")
        if os.path.exists(file):
            state = torch.load(file, map_location=self.device)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optim'])
            self.scheduler.load_state_dict(state['scheduler'])
            
            self.output_dir = ckpt_dir
            self.logger.print("Successfully loaded model checkpoint", mode="info")
        else:
            raise FileNotFoundError(f"Could not find checkpoint at {ckpt_dir}")
        
    def process_state(self, state):
        state = torch.from_numpy(state) / 255.0
        return state.to(self.device)
        
    def initialize_memory(self):
        state = self.process_state(self.env.reset())
        
        for step in range(self.args.memory_size):
            with torch.no_grad():
                action = self.agent.select_action(state, train=False)
            next_frames, reward, done, _ = self.env.step(action)
            next_state = self.process_state(next_frames)
            
            self.memory.add_sample(state, action, next_state, reward, done)
            state = next_state if not done else self.process_state(self.env.reset())            
            utils.progress_bar(progress=(step+1)/self.args.memory_size, desc="Initializing memory", status="")
    
    def train_step(self, batch):
        states, actions = batch[0].to(self.device), batch[1].to(self.device)
        output, _ = self.model(states)
        loss = F.cross_entropy(output, actions)
        acc = output.argmax(-1).eq(actions).float().mean().item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc 
    
    @torch.no_grad()
    def eval_step(self, batch):
        states, actions = batch[0].to(self.device), batch[1].to(self.device)
        output, _ = self.model(states)
        loss = F.cross_entropy(output, actions)
        acc = output.argmax(-1).eq(actions).float().mean().item()
        return loss.item(), acc
        
    def train(self):
        self.initialize_memory()
        self.logger.print("Beginning training", mode="info")
        
        for epoch in range(self.start_epoch, self.config["train_epochs"]+1):
            self.agent.train()
            train_meter = utils.AverageMeter()
            desc = "[TRAIN] Epoch {:3d}/{:3d}".format(epoch, self.config["train_epochs"])
            
            for epoch in range(self.args.train_epochs):
                for step in range(self.args.train_steps_per_epoch):
                    batch = self.memory.get_batch(self.args.batch_size)
                    loss, acc = self.train_step(batch)
                    train_meter.add({'train loss': loss, 'train accuracy': acc})
                    utils.progress_bar((step+1)/self.args.train_steps_per_epoch, desc, train_meter.return_msg())

                if self.args.log_wandb:
                    wandb.log({"Epoch": epoch, **train_meter.return_dict()})
                self.logger.write("Epoch {:3d}/{:3d} {}".format(
                    epoch, self.config["train_epochs"], train_meter.return_msg()
                ), mode="train")
                
                # Eval loop
                if epoch % self.config["eval_every"] == 0:
                    self.agent.eval()
                    val_meter = utils.AverageMeter()
                    desc = "{}[VALID] Epoch {:3d}/{:3d}{}".format(
                        utils.COLORS["blue"], epoch, self.args.train_epochs, utils.COLORS["end"]
                    )
                    for step in range(self.args.eval_steps_per_epoch):
                        batch = self.memory.get_batch(self.args.batch_size)
                        loss, acc = self.eval_step(batch)
                        val_meter.add({'val loss': loss, 'val accuracy': acc})
                        utils.progress_bar((step+1)/self.args.eval_steps_per_epoch, desc, val_meter.return_msg())

                    avg_metrics = val_meter.return_dict()
                    if self.args.log_wandb:
                        wandb.log({"Epoch": epoch, **val_meter})
                    self.logger.record("Epoch {:3d}/{:3d} {}".format(
                        epoch, self.config["train_epochs"], val_meter.return_msg()
                    ), mode="val")
                    
                    if avg_metrics["val accuracy"] > self.best_acc:
                        self.best_acc = avg_metrics["val accuracy"]
                        self.save_checkpoint() 
                        
                # Refresh replay memory every few epochs 
                if epoch % self.args.mem_refresh_interval == 0:
                    self.initialize_memory()
                
        print()
        self.logger.print("Completed training", mode="info")
        
    @torch.no_grad()
    def visualize_attn(self):
        episode_finished = False 
        total_reward = 0
        self.model.eval()
        
        all_states, all_attn_probs = [], []
        state = self.env.reset()
        
        while not episode_finished:
            output, attn_probs = self.model(self.process_state(state))
            action = output.argmax(-1).item()
            all_attn_probs.append(attn_probs[self.args.num_layers-1][0])
            all_states.append(state[0])
            
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if not done:
                state = next_state
            else:
                episode_finished = True     
                
        # Generate attention visualization
        os.makedirs(os.path.join(self.output_dir, 'attn_viz'), exist_ok=True)
        
        for i, (state, attn_prob) in enumerate(zip(all_states, all_attn_probs)):
            plt.figure(figsize=(4 * (self.args.num_heads+1), 4))
            
            for j in range(self.args.num_heads+1):    
                if j == 0:
                    plt.subplot(1, self.args.num_heads+1, 1)
                    plt.imshow(state[-1, :, :], cmap='gray')
                    plt.axis('off')
                else:
                    attn = attn_prob[j-1, 0, 1:].detach().cpu().numpy().reshape(self.h_patches, self.w_patches)
                    attn = cv2.resize(attn, (state.shape[2], state.shape[1]), cv2.INTER_AREA)
                    plt.subplot(1, self.args.num_heads+1, j+1)
                    plt.imshow(attn, cmap='plasma', vmin=attn.min(), vmax=attn.max())
                    plt.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'attn_viz', f'{i}.png'))
            plt.close()
        
        return total_reward