import os
import cv2
import jax 
import math
import utils
import wandb
import optax 
import pickle
import argparse 
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from networks import *
from data_utils import *
from datetime import datetime as dt
from matplotlib import pyplot as plt
from matplotlib import animation as animation

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def main(args):
    
    rng = jax.random.PRNGKey(args.seed)
    
    states, actions = load_data(args.data_path)
    dset = ExperienceDataset(states, actions)
    
    loader = MultiEpochsDataLoader(
        dset,
        shuffle=False,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=numpy_collate
    )
    
    # To preserve RAM, delete copy of states and actions
    state_shape = states[0].shape
    del states, actions
    
    # Model initialization
    model = ViTModel(
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_actions=args.num_actions,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        attn_dropout_rate=args.attn_dropout_rate
    )
    model_rng, dropout_rng = jax.random.split(rng)
    init_rngs = {'params': model_rng, 'dropout': dropout_rng}
    params = model.init(init_rngs, jnp.ones((1, *state_shape)))
    
    # Load checkpoint 
    if args.load is not None:
        fp = os.path.join(args.load, 'ckpt.pkl')
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                saved = pickle.load(f)
            params = saved['params']
        else:
            raise FileNotFoundError(f'Could not find ckpt.pkl at {args.load}')
        expt_dir = args.load
    else:
        print("{}[WARN] No checkpoint loaded! Using random parameters!{}".format(
            utils.COLORS['red'], utils.COLORS['end']
        ))
        expt_dir = './test'
    
    os.makedirs(os.path.join(expt_dir, 'videos'), exist_ok=True)
    
    @jax.jit
    def forward(params, batch):
        states, actions = batch 
        output, attn_probs = model.apply(params, states, training=False, rngs={'dropout': dropout_rng})
        acc = jnp.mean(jnp.argmax(output, -1) == actions)
        return acc, attn_probs
    
    # Generate visualization frames
    meter = utils.AverageMeter()
    resolution = (state_shape[2], state_shape[1])
    os.makedirs(os.path.join(expt_dir, 'attn_viz'), exist_ok=True)
    
    for step, batch in enumerate(loader):
        states, actions = jnp.expand_dims(jnp.asarray(batch[0]).transpose((0, 2, 3, 1)), 0), jnp.asarray(batch[1])   
        acc, attn_probs = forward(params, (states, actions))
        meter.add({'accuracy': acc})
            
        for i in range(states.shape[0]):
            plt.figure(figsize=(4 * (args.num_heads+1), 4))
            for j in range(args.num_heads+1): 
                            
                # In first frame show the observation
                if j == 0:
                    plt.subplot(1, args.num_heads+1, 1)
                    plt.imshow(states[i, :, :, -1], cmap='gray')
                    plt.axis('off')
                    
                # Then show the attention maps for each head
                else:
                    # The prob of a layer has shape (batch_size, num_heads, 442, 442)
                    # After the operation below, we have selected i-th sample, (j-1)-th head ...
                    # ... and attn probs corresponding to CLS token for remaining 441 tokens
                    attn = attn_probs[args.num_layers-1][i, j-1, 0, 1:].reshape(21, 21)
                    attn = cv2.resize(np.asarray(attn), resolution, cv2.INTER_AREA)
                    
                    plt.subplot(1, args.num_heads+1, j+1)
                    plt.imshow(attn, cmap='plasma', vmin=attn.min(), vmax=attn.max())
                    plt.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(expt_dir, 'attn_viz', f'{step * args.batch_size + i}.png'))
            plt.close()
                    
        utils.progress_bar((i+1)/len(dset), "Generating visualization", meter.return_msg())
    
    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--num_actions', type=int, required=True)
    ap.add_argument('--load', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_split', type=float, default=0.8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--num_heads', type=int, default=1)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--patch_size', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--model_dim', type=int, default=256)
    ap.add_argument('--mlp_hidden_dim', type=int, default=512)
    ap.add_argument('--attn_dropout_rate', type=float, default=0.1)
    args = ap.parse_args()
    
    main(args)