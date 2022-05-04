
import os
import jax 
import wandb
import optax 
import pickle
import argparse 
import flax.linen as nn
import jax.numpy as jnp

from utils import *
from networks import *
from data_utils import *
from datetime import datetime as dt

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def main(args):
    
    rng = jax.random.PRNGKey(args.seed)
    states, actions = load_data(args.data_path)
    train_size = int(len(states) * args.train_split) 
    
    train_states, train_actions = states[:train_size], actions[:train_size]
    test_states, test_actions = states[train_size:], actions[train_size:]
    train_dset = ExperienceDataset(train_states, train_actions)
    test_dset = ExperienceDataset(test_states, test_actions)
    
    # To preserve RAM, delete copy of states and actions
    state_shape = states[0].shape
    del states, actions
    
    # Dataloaders
    trainloader = MultiEpochsDataLoader(
        train_dset, 
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=numpy_collate
    )
    testloader = MultiEpochsDataLoader(
        test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=numpy_collate
    )
    
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
    lr_func = optax.cosine_decay_schedule(
        init_value=args.base_lr, decay_steps=len(trainloader)*args.train_epochs, alpha=1e-10
    )
    optim = optax.adamw(learning_rate=lr_func, b1=0.9, b2=0.999, weight_decay=args.weight_decay)
    state = optim.init(params)
    
    # Logging 
    expt_dir = os.path.join(args.out_dir, dt.now().strftime('%d-%m-%Y_%H-%M'))
    os.makedirs(expt_dir, exist_ok=True)
    logger = Logger(expt_dir)
    logger.record_args(args)
    
    if args.wandb:
        run = wandb.init(project='explaining-drl-with-self-attention')
        logger.write("Wandb run URL: {}".format(run.get_url()))
    
    @jax.jit
    def train_step(params, state, batch):
        states, actions = batch 
        actions = actions.squeeze()
        
        def loss_fn(params):
            output, _ = model.apply(params, states, training=True, rngs={'dropout': dropout_rng})
            logits = nn.log_softmax(output, axis=-1)
            labels = jax.nn.one_hot(actions, args.num_actions)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            acc = jnp.mean(jnp.argmax(logits) == labels)
            return loss, acc
        
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optim.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return loss, acc, params, state
    
    @jax.jit
    def eval_step(params, batch):
        states, actions = batch 
        actions = actions.squeeze()
        
        output, _ = model.apply(params, states, training=False, rngs={'dropout': dropout_rng})
        logits = nn.log_softmax(output, axis=-1)
        labels = jax.nn.one_hot(actions, args.num_actions)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        acc = jnp.mean(jnp.argmax(logits) == labels)
        return loss, acc
    
    def evaluate(dataloader):
        meter = AvgMeter()
        
        for step, batch in enumerate(dataloader):
            loss, acc = eval_step(params, batch)
            meter.add({'loss': loss, 'accuracy': acc.item()})
            pbar(p=(step+1)/len(dataloader), msg=meter.msg())
            
        return meter.get()['loss'], meter.get()['accuracy']
    
    def save(epoch, state, params):
        state = {'params': params, 'optim': state, 'epoch': epoch}
        with open(os.path.join(expt_dir, 'ckpt.pkl'), 'wb') as f:
            pickle.dump(state, f)
            
    # Training 
    best_train_loss, best_train_acc = float('inf'), 0.0
    best_test_loss, best_test_acc = float('inf'), 0.0
    
    logger.print("Beginning training...", mode='info')
    for epoch in range(1, args.train_epochs+1):
        meter = AvgMeter()
        
        for step, batch in enumerate(trainloader):
            loss, acc, params, state = train_step(params, state, batch)
            meter.add({'loss': loss, 'accuracy': acc.item()})
            pbar(p=(step+1)/len(trainloader), msg=meter.msg())
            break
            
        avg_train_loss, avg_train_acc = meter.get()['loss'], meter.get()['accuracy']
        avg_test_loss, avg_test_acc = evaluate(testloader)
        
        best_train_loss = min(best_train_loss, avg_train_loss)
        best_train_acc = max(best_train_acc, avg_train_acc)
        best_test_loss = min(best_test_loss, avg_test_loss)
        
        if avg_test_acc >= best_test_acc:
            save(epoch, state, params)
            best_test_acc = avg_test_acc
            
        logger.record("[{}/{}] Train loss: {:.4f} - Train Acc: {:.4f} - Test loss: {:.4f} - Test Acc: {:.4f}".format(
            epoch, args.train_epochs, avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc
        ), mode='train')
        logger.record("[{}/{}] Best Train Acc: {:.4f} - Best Test Acc: {:.4f}".format(
            epoch, args.train_epochs, best_train_acc, best_test_acc
        ), mode='info')
        print("---------------------------------------------------------------------------------------------------")
        
        if args.wandb:
            wandb.log({
                'Train loss': avg_train_loss, 'Train accuracy': avg_train_acc,
                'Test loss': avg_test_loss, 'Test accuracy': avg_test_acc, 'Epoch': epoch
            })
        
        
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--num_actions', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wandb', action='store_true', default=False)
    ap.add_argument('--out_dir', type=str, default='./out')
    ap.add_argument('--train_split', type=float, default=0.8)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--num_heads', type=int, default=1)
    ap.add_argument('--num_layers', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=4)
    ap.add_argument('--model_dim', type=int, default=512)
    ap.add_argument('--mlp_hidden_dim', type=int, default=2048)
    ap.add_argument('--attn_dropout_rate', type=float, default=0.1)
    ap.add_argument('--base_lr', type=float, default=0.0001)
    ap.add_argument('--weight_decay', type=int, default=1e-06)
    ap.add_argument('--train_epochs', type=int, default=500)
    args = ap.parse_args()
    
    main(args)