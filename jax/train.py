
import os
import jax 
import copy
import wandb
import optax
import pickle
import random
import argparse
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from utils import *
from networks import * 
from envs import envs
from collections import deque
from datetime import datetime as dt

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def normalize(arr, p=2):
    return arr / jnp.linalg.norm(arr, ord=p)


class ReplayMemory:
    
    def __init__(self, capacity, num_actions):
        self.per_group_capacity = capacity // num_actions 
        self.mem_groups = {i: self._init_mem_group() for i in range(num_actions)}
        self.group_sizes = {i: 0 for i in range(num_actions)}
    
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
            jnp.concatenate(states), jnp.array(actions), jnp.array(rewards), jnp.concatenate(next_states), jnp.array(dones)
        )
        
    def _sample_separated(self, batch_size):
        sizes = np.array(list(self.group_sizes.values()))
        indx = [np.random.randint(0, sizes[i], size=(batch_size,)) for i in range(len(sizes))]
        states = []

        for i in range(len(sizes)):
            states.append(jnp.concatenate([self.mem_groups[i]['states'][j] for j in indx[i]]))
            
        return states
    
    def sample(self, batch_size, separated=False):
        if not separated:
            return self._sample_normal(batch_size)
        else:
            return self._sample_separated(batch_size)
            
    
def main(args):
    
    # Initialization
    rng = jax.random.PRNGKey(args.seed)
    state_shape = (args.obs_height, args.obs_width)
    
    # Environment
    if args.env_type == 'atari':
        env = envs.AtariEnv(
            args.env_name, args.frame_stack, True, args.clip_rewards
        )
    elif args.env_type == 'highway':
        env = envs.HighwayEnv(
            args.env_name, state_shape, args.frame_stack, args.frame_skip, args.highway_scaling
        )
    elif args.env_type == 'vizdoom':
        env = envs.VizdoomEnv(
            args.env_name, state_shape, args.frame_skip, args.frame_stack, args.vzd_screen_res, args.vzd_screen_format
        )
    
    # Action specific replay memory
    memory = ReplayMemory(args.mem_capacity, env.num_actions)
    
    # Initialize networks 
    encoder = StateEncoder(args.enc_feature_dim)
    q_network = QNetwork(args.hidden_dim, env.num_actions)
    proj_head = ProjectionHead(args.proj_dim)
    
    # Parameters
    enc_rng, qnet_rng, proj_rng, permute_rng = jax.random.split(rng, 4)
    
    enc_params = encoder.init(enc_rng, jnp.ones((1, *state_shape, args.frame_stack)))
    online_q_params = q_network.init(qnet_rng, jnp.ones((1, args.enc_feature_dim)))
    target_q_params = copy.deepcopy(online_q_params) 
    proj_params = proj_head.init(proj_rng, jnp.ones((1, args.enc_feature_dim)))
    
    # Collecting them in a dict so they can be passed around easily in functions
    params = {'enc': enc_params, 'online_q': online_q_params, 'target_q': target_q_params, 'proj': proj_params}
        
    # Optimizers and model states
    optim = optax.adamw(learning_rate=args.lr, b1=0.9, b2=0.999, weight_decay=args.weight_decay)
    model_state = optim.init(params)    
    
    best_reward = -float('inf')
    epsilon = args.eps_max 
    global_step = 0
    
    # Loading mechanism
    if args.load is not None:
        state = load(args.load)
        params = state['params']
        model_state = state['model_state']
        out_dir = args.load
    else:
        out_dir = os.path.join('outputs', args.env_type, args.env_name, dt.now().strftime('%d-%m-%Y_%H-%M'))  
        os.makedirs(out_dir, exist_ok=True)
          
        logger = Logger(out_dir)
        logger.display_args(args)
        if args.wandb:
            run = wandb.init(project='rl-experiments-similarity')
            logger.write("Wandb run: {}".format(run.get_url()))
    
    def save(params, model_state):
        savedata = {'params': params, 'model_state': model_state}
        with open(os.path.join(out_dir, 'ckpt'), 'wb') as f:
            pickle.dump(savedata, f)
        logger.print("Saved model checkpoint", mode='info')
            
    def load(ckpt_dir):
        fp = os.path.join(ckpt_dir, 'ckpt')
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f'Could not find ckpt at {ckpt_dir}')
        
    def decay_epsilon_linear(step):
        new_eps = args.eps_max - (args.eps_max - args.eps_min) * step / args.eps_decay_steps
        return max(new_eps, args.eps_min)
        
    def select_action(state, params, stochastic=True):
        state_fs = encoder.apply(params['enc'], state)
        
        if stochastic and random.uniform(0, 1) < epsilon:
            action = env.random_action()
        else:
            qvals = q_network.apply(params['online_q'], state_fs)
            action = jnp.argmax(qvals, -1)
        
        return action
    
    @jax.jit
    def update_qnet(state, action, reward, next_state, done, params):
        next_state_fs = encoder.apply(params['enc'], next_state)
        next_state_action = q_network.apply(params['online_q'], next_state_fs).argmax(axis=-1, keepdims=True)
        next_state_qvals = jnp.take_along_axis(q_network.apply(params['target_q'], next_state_fs), next_state_action, -1)
        target_qvals = reward + (1-done) * args.gamma * next_state_qvals.reshape(-1,)
        
        def qnet_loss(params):
            state_fs = encoder.apply(params['enc'], state)
            state_qvals = jnp.take_along_axis(q_network.apply(params['online_q'], state_fs), action.reshape(-1, 1), -1)
            loss = optax.huber_loss(state_qvals.reshape(-1,), target_qvals).mean()
            return loss 
        
        loss, grads = jax.value_and_grad(qnet_loss)(params)
        return loss, grads
        
    @jax.jit
    def update_sim(states, params, permute_rng):
        # 'states' is a list of batch of states indexed by the action
        # taken on them. The size of each batch could vary.
        perm_states = []
        for i in range(len(states)):
            permute_rng, _ = jax.random.split(permute_rng)
            order = jax.random.permutation(permute_rng, len(states[i]))
            perm_states.append(states[i][order])
            
        def contrastive_loss(params, temp):
            total_loss = 0
            state_fs, perm_state_fs = [], []
            
            # Extract all the state vectors and normalize
            for i in range(len(states)):
                state_fs.append(
                    normalize(proj_head.apply(params['proj'], encoder.apply(params['enc'], states[i])), 2)
                )
                perm_state_fs.append(
                    normalize(proj_head.apply(params['proj'], encoder.apply(params['enc'], perm_states[i])), 2)
                )
            
            # Compute loss for each action batch
            for i in range(len(states)):
                pos1, pos2 = state_fs[i], perm_state_fs[i]                                                  # (N_i, d), (N_i, d)
                neg1 = jnp.concatenate([state_fs[j] for j in range(len(state_fs)) if j != i])               # (N', d)
                neg2 = jnp.concatenate([perm_state_fs[j] for j in range(len(perm_state_fs)) if j != i])     # (N', d)
                neg = jnp.concatenate([neg1, neg2], 0)                                                      # (2N', d)
                
                pos1_pos2 = jnp.diag(jnp.matmul(pos1, pos2.T)).reshape(-1, 1) / temp                        # (N_i, 1)
                pos2_pos1 = jnp.diag(jnp.matmul(pos2, pos1.T)).reshape(-1, 1) / temp                        # (N_i, 1)
                pos1_neg = jnp.matmul(pos1, neg.T) / temp                                                   # (N_i, N')
                pos2_neg = jnp.matmul(pos2, neg.T) / temp                                                   # (N_i, N')
                
                scores_1 = jnp.concatenate([pos1_pos2, pos1_neg], axis=1)                                   # (N_i, 1+N')
                scores_2 = jnp.concatenate([pos2_pos1, pos2_neg], axis=1)                                   # (N_i, 1+N')
                scores = jnp.concatenate([scores_1, scores_2])                                              # (2*N_i, 1+N')
                
                logits = nn.log_softmax(scores, -1)
                target = jnp.zeros(logits.shape, dtype=jnp.int32)
                target.at[:, 0].set(1)
                
                loss = optax.softmax_cross_entropy(logits, target).mean()
                total_loss += loss
                
            return total_loss / len(states)
        
        loss, grads = jax.value_and_grad(contrastive_loss)(params, args.sim_temp)
        return loss, grads, permute_rng
    
    def train_episode(params, model_state, epsilon, global_step, permute_rng):
        q_losses, sim_losses = [], []
        total_reward = 0
        state = env.reset()
        
        while True:
            action = select_action(state, params, stochastic=True)
            next_state, reward, done, _ = env.step(action)
            global_step += 1
            epsilon = decay_epsilon_linear(global_step)
            
            memory.add(state, action, reward, next_state, done)
            total_reward += reward
            
            if global_step > 2*args.batch_size and global_step % args.learning_interval == 0:
                q_learning_batch = memory.sample(args.batch_size, separated=False)
                sim_learning_batch = memory.sample(args.batch_size, separated=True)
                
                q_loss, q_grads = update_qnet(*q_learning_batch, params)
                sim_loss, sim_grads, permute_rng = update_sim(sim_learning_batch, params, permute_rng)
                q_losses.append(q_loss), sim_losses.append(sim_loss)
                
                total_grads = jax.tree_map(lambda q_grad, sim_grad: q_grad + args.sim_weight * sim_grad, q_grads, sim_grads)
                updates, model_state = optim.update(total_grads, model_state, params)
                params = optax.apply_updates(params, updates)
                
                # Soft update of target params with online params
                params['target_q'] = jax.tree_map(
                    lambda trg, src: (1 - args.tau) * trg + args.tau * src, params['target_q'], params['online_q']
                )
                
            if not done:
                state = next_state
            else:
                break
            
        metrics = {
            'total_reward': total_reward, 
            'q_loss': np.mean(q_losses) if len(q_losses) > 0 else 0.0, 
            'sim_loss': np.mean(sim_losses) if len(sim_losses) > 0 else 0.0
        }
        return params, model_state, epsilon, global_step, permute_rng, metrics
    
    def eval_episode(params):
        total_reward = 0
        state = env.reset()
        
        while True:
            action = select_action(state, params, stochastic=False)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward 
            
            if not done:
                state = next_state    
            else:
                break
            
        return total_reward
    
    # Training loop
    for epoch in range(1, args.train_epochs+1):
        meter = AvgMeter()
        
        for ep in range(args.train_episodes_per_epoch):
            params, model_state, epsilon, global_step, permute_rng, metrics = train_episode(
                params, model_state, epsilon, global_step, permute_rng
            )
            meter.add(metrics)
            pbar((ep+1)/args.train_episodes_per_epoch, desc=f'Train Epoch [{epoch}/{args.train_epochs}]', status=meter.msg())
        
        logger.write(f"Epoch [{epoch}/{args.train_epochs}] {meter.msg()} [epsilon] {round(epsilon, 4)}", mode='train')
        if args.wandb:
            epoch_avg = meter.avg()
            wandb.log({'epoch': epoch, 'epsilon': epsilon, **{f'train {k}': v for k, v in epoch_avg.items()}})
            
        # Evaluation loop
        if epoch % args.eval_interval == 0:
            ep_rewards = 0
            
            for ep in range(args.eval_episodes_per_epoch):
                reward = eval_episode(params, epsilon)
                ep_rewards += reward
                pbar((ep+1)/args.eval_episodes_per_epoch, desc=f'Val Epoch [{epoch}/{args.train_epochs}]')
               
            mean_reward = ep_rewards / args.eval_episodes_per_epoch
            logger.write(f"Epoch [{epoch}/{args.train_epochs}] [total_reward] {round(mean_reward, 4)}", mode='val')   
            if args.wandb:
                wandb.log({'epoch': epoch, 'eval total_reward': mean_reward})
                
            if mean_reward > best_reward:
                best_reward = mean_reward
                save(params, model_state)
                
                
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wandb', action='store_true', default=False)
    ap.add_argument('--load', type=str, default=None)
    
    # Environment
    ap.add_argument('--obs_height', type=int, default=84)
    ap.add_argument('--obs_width', type=int, default=84)
    ap.add_argument('--env_type', type=str, required=True)
    ap.add_argument('--env_name', type=str, required=True)
    ap.add_argument('--frame_stack', type=int, default=4)
    ap.add_argument('--frame_skip', type=int, default=4)
    ap.add_argument('--clip_rewards', action='store_true', default=False)
    ap.add_argument('--highway_scaling', type=float, default=1.5)
    ap.add_argument('--vzd_screen_res', type=str, default='RES_640X480')
    ap.add_argument('--vzd_screen_format', type=str, default='GRAY8')
    
    # Replay memory
    ap.add_argument('--mem_capacity', type=int, default=100000)
    ap.add_argument('--batch_size', type=int, default=32)
    
    # Agent architecture
    ap.add_argument('--enc_feature_dim', type=int, default=1024)
    ap.add_argument('--hidden_dim', type=int, default=512)
    ap.add_argument('--proj_dim', type=int, default=128)
    
    # Training
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--weight_decay', type=float, default=1e-05)
    ap.add_argument('--eps_max', type=float, default=1.0)
    ap.add_argument('--eps_min', type=float, default=0.01)
    ap.add_argument('--eps_decay_steps', type=int, default=5000000)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--tau', type=float, default=0.005)
    ap.add_argument('--sim_temp', type=float, default=0.1)
    ap.add_argument('--sim_weight', type=float, default=2.0)
    ap.add_argument('--train_epochs', type=int, default=100)
    ap.add_argument('--train_episodes_per_epoch', type=int, default=1000)
    ap.add_argument('--eval_interval', type=int, default=5)
    ap.add_argument('--eval_episodes_per_epoch', type=int, default=100)
    ap.add_argument('--learning_interval', type=int, default=4)
    
    args = ap.parse_args()
    
    # Run
    main(args)