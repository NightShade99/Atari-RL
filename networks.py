
import jax
import jax.numpy as jnp 
import flax.linen as nn


__all__ = ['ViTModel', 'ViTLayer', 'MultiHeadSelfAttention', 'unfold_img_to_sequence']


def unfold_img_to_sequence(inp, patch_size):
    assert inp.shape[1] % patch_size == 0, f'Input height {inp.shape[1]} not divisible by {patch_size}'
    assert inp.shape[2] % patch_size == 0, f'Input width {inp.shape[2]} not divisible by {patch_size}'
    
    sequence = []
    bs, h, w, _ = inp.shape
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            p = inp[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :].reshape(bs, -1)
            sequence.append(p)            
    
    return jnp.stack(sequence, 1)


class MultiHeadSelfAttention(nn.Module):
    num_heads: int = 1
    model_dim: int = 512
    dropout_rate: float = 0.1
    
    def setup(self):
        kernel_init = nn.initializers.xavier_normal() 
        
        self.layernorm = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.query = nn.Dense(self.model_dim, use_bias=False, kernel_init=kernel_init)
        self.key = nn.Dense(self.model_dim, use_bias=False, kernel_init=kernel_init)
        self.value = nn.Dense(self.model_dim, use_bias=False, kernel_init=kernel_init)
        
    def __call__(self, x, training=True):
        bs, sl, _ = x.shape 
        head_dim = self.model_dim // self.num_heads
        
        x_norm = self.layernorm(x)
        q = self.query(x_norm).reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        k = self.key(x_norm).reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        v = self.value(x_norm).reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        
        attn_scores = jnp.einsum('bhid,bhjd->bhij', q, k)
        attn_probs = nn.softmax(attn_scores / jnp.sqrt(head_dim), axis=-1)
        attn_probs = self.dropout(attn_probs, deterministic=(not training))
        
        out = jnp.einsum('bhij,bhjd->bhid', attn_probs, v)
        out = out.transpose((0, 2, 1, 3)).reshape(bs, sl, self.model_dim)
        return out + x, attn_probs
    
    
class ViTLayer(nn.Module):
    num_heads: int = 1
    model_dim: int = 512
    mlp_hidden_dim: int = 2048 
    attn_dropout_rate: float = 0.1
    
    def setup(self):
        kernel_init = nn.initializers.xavier_normal()
        bias_init = nn.initializers.normal(stddev=1e-06)
        
        self.attention = MultiHeadSelfAttention(self.num_heads, self.model_dim, self.attn_dropout_rate)
        self.mlp_fc1 = nn.Dense(self.mlp_hidden_dim, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_fc2 = nn.Dense(self.model_dim, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_layernorm = nn.LayerNorm()
        
    def __call__(self, x, training=True):
        x, attn_probs = self.attention(x, training)
        x_norm = self.mlp_layernorm(x)
        x_norm = self.mlp_fc2(nn.gelu(self.mlp_fc1(x_norm)))
        return x_norm + x, attn_probs
    
    
class ViTModel(nn.Module):
    num_actions: int
    num_heads: int = 1
    num_layers: int = 4
    patch_size: int = 4
    model_dim: int = 512
    mlp_hidden_dim: int = 2048
    attn_dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, inp, training=True):
        # Convert image to sequence of flattened patches
        x = unfold_img_to_sequence(inp, self.patch_size)
        bs, seqlen, inp_fdim = x.shape 
         
        # Append CLS token and scale features up to model_dim
        cls_token = self.param('cls', nn.initializers.zeros, (bs, 1, inp_fdim))
        x = jnp.concatenate([cls_token, x], axis=1)
        x = nn.Dense(self.model_dim)(x)
        
        # Add positional embeddings
        pos_embeds = nn.Embed(seqlen+1, self.model_dim)(jnp.arange(seqlen+1))
        x = x + pos_embeds 
        
        # Pass through self attention layers 
        layerwise_attn_probs = {}
        for i in range(self.num_layers):
            x, attn_probs = ViTLayer(self.num_heads, self.model_dim, self.mlp_hidden_dim, self.attn_dropout_rate)(x, training)
            layerwise_attn_probs[i] = attn_probs
            
        # Classifier
        preds = nn.Dense(self.num_actions)(x[:, 0, :])
        
        return preds, layerwise_attn_probs
    
    
# if __name__ == '__main__':
    
#     model = ViTModel(num_heads=1, num_layers=4, patch_size=4, model_dim=512, mlp_hidden_dim=2048, attn_dropout_rate=0.1)
#     rng = jax.random.PRNGKey(0)
#     params_rng, dropout_rng = jax.random.split(rng)
    
#     init_rngs = {'params': params_rng, 'dropout': dropout_rng}
#     variables = model.init(init_rngs, jnp.ones((1, 84, 84, 4)))

#     x = jax.random.normal(rng, (1, 84, 84, 4))
#     out = model.apply(variables, x, training=True, rngs={'dropout': dropout_rng})    
#     # print(out[0].shape)
    
#     print(out[1][0][0, 0, :5, :5])