
import os 
import cv2
import jax 
import utils
import optax
import wandb 
import torch
import pickle
import argparse
import numpy as np 
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt

from networks import *
from envs.envs import *
from datetime import datetime as dt


class ReplayMemory:
    
    def __init__(self, memory_size, stack_size=4):
        self.ptr = 0
        self.filled = 0
        self.stack_size = stack_size
        self.mem_size = memory_size
        
        self.states = np.zeros((self.mem_size, self.stack_size, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((self.mem_size, self.stack_size, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((self.mem_size,), dtype=np.uint8)
        self.rewards = np.zeros((self.mem_size,), dtype=np.float32)
        self.terminal = np.zeros((self.mem_size,), dtype=np.uint8) 
            
    def add_sample(self, state, action, next_state, reward, done):
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
        
        state = jnp.asarray(self.states[idx] / 255.0)
        next_state = jnp.asarray(self.next_states[idx] / 255.0)
        action = jnp.asarray(self.actions[idx])
        reward = jnp.asarray(self.rewards[idx])
        done = jnp.asarray(self.terminal[idx])
        return state, action, next_state, reward, done
    
    
def load_checkpoint(ckpt_dir, device):
    file = os.path.join(ckpt_dir, "checkpoint.pt")
    if os.path.exists(file):
        return torch.load(file, map_location=device)
    else:
        raise FileNotFoundError(f"Could not find checkpoint at {ckpt_dir}")

def save_checkpoint(epoch, state, params, output_dir):
    state = {'params': params, 'optim': state, 'epoch': epoch}
    with open(os.path.join(output_dir, 'ckpt.pkl'), 'wb') as f:
        pickle.dump(state, f)


def main(args):
    
    rng = jax.random.PRNGKey(0)
    config, output_dir, logger, device = utils.initialize_experiment(
        args, output_root=f'outputs/attention/{args.env_type}/{args.env_name}', ckpt_dir=args.load
    )
    if args.log_wandb:
        run = wandb.init(project="atari-experiments")
        logger.write("Wandb url: {}".format(run.get_url()), mode="info")
    
    start_epoch = 1
    best_acc = 0
    args = args
        
    if args.env_type == 'atari':
        env = AtariEnv(args.env_name, **config['environment'])
    elif args.env_type == 'highway':
        env = HighwayEnv(args.env_name, **config['environment'])
    elif args.env_type == 'vizdoom':
        env = VizdoomEnv(args.env_name, **config['environment'])
 
    assert args.load is not None, f'Agent checkpoint needed for attention training'            
    agent = load_checkpoint(args.load, device)
    memory = ReplayMemory(args.memory_size, config['environment']['frame_stack'])        
    state_shape = (84, 84, config['environment']['frame_stack'])
    
    # Attention model
    model = ViTModel(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_actions=env.num_actions,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        attn_dropout_rate=args.attn_dropout_rate
    )
    model_rng, dropout_rng = jax.random.split(rng)
    init_rngs = {'params': model_rng, 'dropout': dropout_rng}
    
    params = model.init(init_rngs, jnp.ones((args.batch_size, *state_shape)))
    lr_func = optax.cosine_decay_schedule(
        init_value=args.lr, decay_steps=args.train_steps_per_epoch * args.train_epochs, alpha=1e-10
    )
    optim = optax.adamw(learning_rate=lr_func, b1=0.9, b2=0.999, weight_decay=args.weight_decay)
    model_state = optim.init(params)
    
    def process_state(state):
        state = torch.from_numpy(state / 255.0).float()
        return state.to(device)
    
    def initialize_memory():
        state = env.reset()
        
        for step in range(args.memory_size):
            action = agent.select_action(process_state(state), train=False)
            next_state, reward, done, _ = env.step(action)
            
            memory.add_sample(state, action, next_state, reward, done)
            state = next_state if not done else env.reset()            
            utils.progress_bar(progress=(step+1)/args.memory_size, desc="Initializing memory", status="")
    
    @jax.jit
    def train_step(params, state, batch):
        states, actions = batch[0], batch[1] 
        states = jnp.transpose(states, (0, 2, 3, 1))
        
        def loss_fn(params):
            output, _ = model.apply(params, states, training=True, rngs={'dropout': dropout_rng})
            logits = nn.log_softmax(output, axis=-1)
            labels = jax.nn.one_hot(actions, env.num_actions)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            acc = jnp.mean(jnp.argmax(logits, -1) == actions)
            return loss, acc
        
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optim.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return loss, acc, params, state
    
    @jax.jit
    def eval_step(params, batch):
        states, actions = batch[0], batch[1]
        states = jnp.transpose(states, (0, 2, 3, 1))
        
        output, _ = model.apply(params, states, training=False, rngs={'dropout': dropout_rng})
        logits = nn.log_softmax(output, axis=-1)
        labels = jax.nn.one_hot(actions, env.num_actions)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        acc = jnp.mean(jnp.argmax(logits, -1) == actions)
        return loss, acc
            
    # Training
    initialize_memory()
    logger.print("Beginning training", mode="info")
    
    for epoch in range(start_epoch, args.train_epochs+1):
        agent.train()
        train_meter = utils.AverageMeter()
        desc = "[TRAIN] Epoch {:3d}/{:3d}".format(epoch, args.train_epochs)
        
        for step in range(args.train_steps_per_epoch):
            batch = memory.get_batch(args.batch_size)
            loss, acc, params, model_state = train_step(params, model_state, batch)
            train_meter.add({'train loss': loss, 'train accuracy': acc})
            utils.progress_bar((step+1)/args.train_steps_per_epoch, desc, train_meter.return_msg())

        if args.log_wandb:
            wandb.log({"Epoch": epoch, **train_meter.return_dict()})
        logger.write("Epoch {:3d}/{:3d} {}".format(
            epoch, config["train_epochs"], train_meter.return_msg()
        ), mode="train")
        
        # Refresh replay memory every few epochs 
        if epoch != args.train_epochs and epoch % args.mem_refresh_interval == 0:
            initialize_memory()
        
        # Eval loop
        if epoch % config["eval_every"] == 0:
            agent.eval()
            val_meter = utils.AverageMeter()
            desc = "{}[VALID] Epoch {:3d}/{:3d}{}".format(
                utils.COLORS["blue"], epoch, args.train_epochs, utils.COLORS["end"]
            )
            for step in range(args.eval_steps_per_epoch):
                batch = memory.get_batch(args.batch_size)
                loss, acc = eval_step(params, batch)
                val_meter.add({'val loss': loss, 'val accuracy': acc})
                utils.progress_bar((step+1)/args.eval_steps_per_epoch, desc, val_meter.return_msg())

            avg_metrics = val_meter.return_dict()
            if args.log_wandb:
                wandb.log({"Epoch": epoch, **avg_metrics})
            logger.record("Epoch {:3d}/{:3d} {}".format(
                epoch, config["train_epochs"], val_meter.return_msg()
            ), mode="val")
            
            if avg_metrics["val accuracy"] > best_acc:
                best_acc = avg_metrics["val accuracy"]
                save_checkpoint(epoch, model_state, params, output_dir) 
                    
    print()
    logger.print("Completed training", mode="info")
    
    # Create a small dataset for visualization
    os.makedirs(f'datasets/{args.env_name}', exist_ok=True)
    states, actions = [], []
    state = env.reset()
    
    for step in range(1000):
        action = agent.select_action(process_state(state), train=False)
        states.append(state), actions.append(action)

        next_state, _, done, _ = env.step(action)
        state = process_state(next_state if not done else env.reset())        
        utils.progress_bar(progress=(step+1)/1000, desc="Visualization dataset", status="")
            
    savedata = {'states': np.concatenate(states, 0), 'actions': np.array(actions)}
    with open(f'datasets/{args.env_name}/data.pkl', 'wb') as f:
        pickle.dump(savedata, f)
        

if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--log_wandb', action='store_true', default=False)
    ap.add_argument('--config', required=True)
    ap.add_argument('--env_name', required=True, type=str)
    ap.add_argument('--env_type', required=True, choices=["atari", "vizdoom", "highway"], type=str)
    ap.add_argument('--task', required=True, choices=["train", "anim"], type=str)
    ap.add_argument('--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str)
    ap.add_argument('--dset_save_dir', default='./datasets', type=str)
    
    # Attention training args
    ap.add_argument('--patch_size', default=4, type=int)
    ap.add_argument('--batch_size', default=128, type=int)
    ap.add_argument('--num_layers', default=2, type=int)
    ap.add_argument('--num_heads', default=1, type=int)
    ap.add_argument('--model_dim', default=256, type=int)
    ap.add_argument('--mlp_hidden_dim', default=512, type=int)
    ap.add_argument('--attn_dropout_rate', default=0.1, type=float)
    ap.add_argument('--lr', default=0.0001, type=float)
    ap.add_argument('--weight_decay', default=1e-06, type=float)
    ap.add_argument('--train_epochs', default=100, type=int)
    ap.add_argument('--memory_size', default=1000, type=int)
    ap.add_argument('--train_steps_per_epoch', default=1000, type=int)
    ap.add_argument('--eval_steps_per_epoch', default=1000, type=int)
    ap.add_argument('--mem_refresh_interval', default=5, type=int)
    
    args = ap.parse_args()
    
    main(args)
