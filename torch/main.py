
import os 
import trainers 
import argparse 
from datetime import datetime as dt


if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--log_wandb', action='store_true', default=False)
    ap.add_argument('--config', required=True)
    ap.add_argument('--env_name', required=True, type=str)
    ap.add_argument('--env_type', required=True, choices=["atari", "vizdoom", "highway"], type=str)
    ap.add_argument('--task', required=True, type=str)
    ap.add_argument('--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str)
    ap.add_argument('--dset_save_dir', default='./datasets', type=str)
    ap.add_argument('--num_samples', default=1_000_000, type=int)
    
    # Attention training args
    ap.add_argument('--memory_size', default=1000, type=int)
    ap.add_argument('--patch_size', default=4, type=int)
    ap.add_argument('--batch_size', default=64, type=int)
    ap.add_argument('--num_layers', default=2, type=int)
    ap.add_argument('--num_heads', default=1, type=int)
    ap.add_argument('--model_dim', default=256, type=int)
    ap.add_argument('--mlp_hidden_dim', default=512, type=int)
    ap.add_argument('--attn_dropout_rate', default=0.1, type=float)
    ap.add_argument('--lr', default=0.0001, type=float)
    ap.add_argument('--weight_decay', default=1e-06, type=float)
    ap.add_argument('--train_epochs', default=500, type=int)
    ap.add_argument('--train_steps_per_epoch', default=100, type=int)
    ap.add_argument('--eval_steps_per_epoch', default=100, type=int)
    ap.add_argument('--mem_refresh_interval', default=5, type=int)
    
    args = ap.parse_args()
    
    
    if args.task == 'train':
        trainer = trainers.Trainer(args)
        trainer.train()
        
    if args.task == 'attn_train':
        trainer = trainers.AttentionTrainer(args)
        trainer.train()
        
    elif args.task == "anim":
        trainer = trainers.Trainer(args)
        assert args.load is not None, "Load a model for inference tasks"
        trainer.create_animation()
        
    elif args.task == "dset":
        trainer = trainers.Trainer(args)
        assert args.load is not None, "Load a model for inference tasks"
        trainer.collect_experience()
        
    elif args.task == 'attn':
        trainer = trainers.AttentionTrainer(args)
        assert args.load is not None, "Load a model for inference tasks"
        trainer.visualize_attn()