
from sre_parse import State
import jax
import jax.numpy as jnp
import flax.linen as nn

__all__ = ['StateEncoder', 'QNetwork', 'ProjectionHead']


class StateEncoder(nn.Module):
    feature_dim: int 
    
    @nn.compact 
    def __call__(self, x):
        x = nn.Conv(32, (8, 8), (4, 4), padding='VALID', use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), (2, 2), padding='VALID', use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (1, 1), padding='VALID', use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Conv(self.feature_dim, (7, 7), (1, 1), padding='VALID', use_bias=False)(x)
        x = nn.relu(x)
        x = x.reshape(-1, self.feature_dim)
        return x
    

class QNetwork(nn.Module):
    hidden_dim: int 
    num_actions: int
    
    @nn.compact 
    def __call__(self, x):
        x_value, x_action = jnp.split(x, 2, -1)
        
        x_value = nn.Dense(self.hidden_dim)(x_value)
        x_value = nn.relu(x_value)
        x_value = nn.Dense(1)(x_value)
        
        x_action = nn.Dense(self.hidden_dim)(x_action)
        x_action = nn.relu(x_action)
        x_action = nn.Dense(self.num_actions)(x_action)
        
        q_vals = x_value + (x_action - x_action.mean(-1, keepdims=True))
        return q_vals
    
    
class ProjectionHead(nn.Module):
    out_dim: int 
    
    @nn.compact 
    def __call__(self, x):
        x = nn.Dense(self.out_dim)(x)
        x = nn.relu(x) 
        x = nn.Dense(self.out_dim)(x)
        return x
    
    
# if __name__ == '__main__':
    
#     import optax
#     import flax.linen as nn 
    
#     k = jax.random.PRNGKey(0)
#     enc = StateEncoder(1024)
#     qnet = QNetwork(512, 10)
    
#     enc_params = enc.init(k, jnp.ones((1, 84, 84, 4)))
#     q_params = qnet.init(k, jnp.ones((1, 1024)))
    
#     optim = optax.adamw(learning_rate=0.01, weight_decay=1e-06)
#     state = optim.init({'enc': enc_params, 'qnet': q_params})
    
#     @jax.jit
#     def update(state):
#         x = jax.random.uniform(k, (32, 84, 84, 4))
#         fs = enc.apply(enc_params, x)
        
#         def loss(params):
#             out = nn.log_softmax(qnet.apply(params['qnet'], fs), -1)
#             trg = jax.nn.one_hot(jnp.zeros((32,)), num_classes=10)
#             loss = optax.softmax_cross_entropy(out, trg).mean()
#             return loss 
        
#         loss, grads = jax.value_and_grad(loss)({'enc': enc_params, 'qnet': q_params})
#         updates, state = optim.update(grads, state, {'enc': enc_params, 'qnet': q_params})
#         new_params = optax.apply_updates(updates, {'enc': enc_params, 'qnet': q_params})
#         return new_params
    
#     new_params = update(state)
#     new_enc_params, new_q_params = new_params.values()
    
#     print("Encoder params:", jax.tree_map(lambda x, y: x == y, enc_params, new_enc_params))
#     print("QNet params:", jax.tree_map(lambda x, y: x == y, q_params, new_q_params)) 