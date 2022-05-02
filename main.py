
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
    args = ap.parse_args()

    if args.task == 'train':
        trainer = trainers.Trainer(args)
        trainer.train()
        
    elif args.task == "anim":
        assert args.load is not None, "Load a model for inference tasks"
        trainer = trainers.Trainer(args)
        trainer.create_animation()