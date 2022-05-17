
import os
import jax 
import copy
import optax
import pickle
import random
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from networks import * 
from collections import deque


def normalize(arr, p=2):
    return arr / jnp.linalg.norm(arr, ord=p)


class ReplayMemory:
    
    def __init__(self, rng, capacity):
        self.states = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.rng = rng
        
    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def sample(self, batch_size):
        self.rng, _ = jax.random.split(self.rng)
        idx = jax.random.randint(self.rng, (batch_size,), 0, len(self.states))
        
        states = jnp.stack([self.states[i] for i in idx])
        actions = jnp.stack([self.actions[i] for i in idx])
        rewards = jnp.stack([self.rewards[i] for i in idx])
        next_states = jnp.stack([self.next_states[i] for i in idx])
        dones = jnp.stack([self.dones[i] for i in idx])
        
        return states, actions, rewards, next_states, dones
    
    
def main(args):
    
    # Initialization
    rng = jax.random.PRNGKey(args.seed)
    state_shape = (args.frame_stack, args.obs_height, args.obs_width)
    out_dir = None      # TODO
    
    # Environment and replay memory
    env = None          # TODO

    # Dict of action specific replay memory
    memory = {}
    for i in range(env.num_actions):
        rng, _ = jax.random.split(rng)
        memory[i] = ReplayMemory(rng, args.capacity // env.num_actions, args.batch_size)
    
    # Initialize networks 
    encoder = StateEncoder(args.enc_feature_dim)
    q_network = QNetwork(args.hidden_dim, env.num_actions)
    proj_head = ProjectionHead(args.proj_dim)
    
    # Parameters
    enc_rng, qnet_rng, proj_rng, permute_rng = jax.random.split(rng, 4)
    enc_params = encoder.init(enc_rng, jnp.ones((1, *state_shape)))
    online_q_params = q_network.init(qnet_rng, jnp.ones((1, args.enc_feature_dim)))
    target_q_params = copy.deepcopy(online_q_params) 
    proj_params = proj_head.init(proj_rng, jnp.ones((1, args.enc_feature_dim)))
        
    # Optimizers and model states
    qnet_optim = optax.adamw(learning_rate=args.qnet_lr, b1=0.9, b2=0.999, weight_decay=args.qnet_weight_decay)
    proj_optim = optax.adamw(learning_rate=args.proj_lr, b1=0.9, b2=0.999, weight_decay=args.proj_weight_decay)
    
    qnet_state = qnet_optim.init({'enc': enc_params, 'qnet': online_q_params})
    proj_state = proj_optim.init({'enc': enc_params, 'proj': proj_params})
    
    # Other variables 
    global_step = 0
    epsilon = args.eps_max 
    gamma = args.gamma 
    tau = args.tau
    
    # Loading mechanism
    if args.load is not None:
        state = load(args.load)
        enc_params = state['enc_params']
        online_q_params = state['online_q_params']
        target_q_params = state['target_q_params']
        proj_params = state['proj_params']
        qnet_state = state['qnet_state']
        proj_state = state['proj_state']
    
    def save(enc_params, online_q_params, target_q_params, proj_params, qnet_state, proj_state):
        state = {
            'enc_params': enc_params,
            'online_q_params': online_q_params,
            'target_q_params': target_q_params,
            'proj_params': proj_params,
            'qnet_state': qnet_state,
            'proj_state': proj_state
        }
        with open(os.path.join(out_dir, 'ckpt'), 'wb') as f:
            pickle.dump(state, f)
            
    def load(ckpt_dir):
        fp = os.path.join(ckpt_dir, 'ckpt')
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                state = pickle.load(f)
            return state
        else:
            raise FileNotFoundError(f'Could not find ckpt at {ckpt_dir}')
        
    def select_action(state, stochastic=True):
        state_fs = encoder.apply(enc_params, state)
        if stochastic and random.uniform(0, 1) < epsilon:
            action = env.random_action()
        else:
            qvals = q_network.apply(online_q_params, state_fs)
            action = jnp.argmax(qvals, -1)
            
        global_step += 1
        return action
    
    @jax.jit
    def update_qnet(state, action, reward, next_state, done, qnet_state):
        next_state_fs = encoder.apply(enc_params, next_state)
        next_state_action = q_network.apply(online_q_params, next_state_fs).argmax(axis=-1, keepdims=True)
        next_state_qvals = jnp.take_along_axis(q_network.apply(target_q_params, next_state_fs), next_state_action, -1)
        target_qvals = reward + (1-done) * gamma * next_state_qvals.reshape(-1,)
        
        def qnet_loss(params):
            state_fs = encoder.apply(params['enc'], state)
            state_qvals = jnp.take_along_axis(q_network.apply(params['qnet'], state_fs), action.reshape(-1, 1), -1)
            loss = optax.huber_loss(state_qvals.reshape(-1,), target_qvals).mean()
            return loss 
        
        params = {'enc': enc_params, 'qnet': online_q_params}
        loss, grads = jax.value_and_grad(qnet_loss)(params)
        updates, qnet_state = qnet_optim.update(grads, qnet_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, qnet_state 
        
    @jax.jit
    def update_sim(states, proj_state):
        # 'states' is a list of batch of states indexed by the action
        # taken on them. The size of each batch could vary.
        perm_states = []
        for i in range(len(states)):
            permute_rng, _ = jax.random.split(permute_rng)
            order = jax.random.permutation(permute_rng, len(states[i]))
            perm_states.append(states[i][order])
            
        def contrastive_loss(params, temp):
            losses, state_fs, perm_state_fs = [], [], []
            
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
                pos1, pos2 = state_fs[i], perm_state_fs[i]                                              # (N_i, d), (N_i, d)
                neg1 = jnp.concatenate([states[j] for j in range(len(states)) if j != i])               # (N', d)
                neg2 = jnp.concatenate([perm_states[j] for j in range(len(perm_states)) if j != i])     # (N', d)
                neg = jnp.concatenate([neg1, neg2])                                                     # (2N', d)
                mask = jnp.eye(pos1.shape[0], dtype=bool)
                
                pos1_pos2 = jnp.matmul(pos1, pos2.T)[mask].reshape(-1, 1) / temp                        # (N_i, 1)
                pos2_pos1 = jnp.matmul(pos2, pos1.T)[mask].reshape(-1, 1) / temp                        # (N_i, 1)
                pos1_neg = jnp.matmul(pos1, neg.T) / temp                                               # (N_i, N')
                pos2_neg = jnp.matmul(pos2, neg.T) / temp                                               # (N_i, N')
                
                scores_1 = jnp.concatenate([pos1_pos2, pos1_neg], axis=1)                               # (N_i, 1+N')
                scores_2 = jnp.concatenate([pos2_pos1, pos2_neg], axis=1)                               # (N_i, 1+N')
                scores = jnp.concatenate([scores_1, scores_2])                                          # (2*N_i, 1+N')
                
                logits = nn.log_softmax(scores, -1)
                target = jnp.zeros(logits.shape, dtype=jnp.int32)
                target[:, 0] = 1
                
                loss = optax.softmax_cross_entropy(logits, target).mean()
                losses.append(loss)
                
            return jnp.mean(losses)
        
        params = {'enc': enc_params, 'proj': proj_params}
        loss, grads = jax.value_and_grad(contrastive_loss)(params, args.contrastive_loss_temp)
        updates, proj_state = proj_optim.update(grads, proj_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, proj_state
    
    def train_episode():
        rewards, q_losses, sim_losses = [], [], []
        state = env.reset()
        
        while True:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            
            memory[action].add(state, action, reward, next_state, done)
            
            if global_step % args.q_learning_interval == 0:
                batch = memory.sample(args.batch_size, separate=False)
                loss, new_params, qnet_state = update_qnet(*batch, qnet_state)
                enc_params, online_q_params = new_params['enc'], new_params['qnet']
                q_losses.append(loss)
                
            if global_step % args.sim_learning_interval == 0:
                batches = memory.sample(args.batch_size, separate=True)
                loss, new_params, proj_state = update_sim(batches, proj_state)
                enc_params, proj_params = new_params['enc'], new_params['proj']
                sim_losses.append(loss)
                
            if not done:
                state = next_state
            else:
                break
            
        return np.mean(rewards), np.mean(q_losses), np.mean(sim_losses)  
    
    