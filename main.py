
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
    ap.add_argument('--task', required=True, choices=["train", "anim"], type=str)
    ap.add_argument('--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str)
    ap.add_argument('--dset_save_dir', default='./datasets', type=str)
    ap.add_argument('--num_samples', default=1_000_000, type=int)
    args = ap.parse_args()
    
    trainer = trainers.Trainer(args)

    if args.task == 'train':
        trainer.train()
        
    elif args.task == "anim":
        assert args.load is not None, "Load a model for inference tasks"
        trainer.create_animation()
        
    elif args.task == "dset":
        assert args.load is not None, "Load a model for inference tasks"
        trainer.collect_experience()