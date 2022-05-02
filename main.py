
import os 
import trainers 
import argparse 
from datetime import datetime as dt


if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--load', default=None)
    ap.add_argument('-r', '--resume', default=None)
    ap.add_argument('-r', '--log_wandb', action='store_true', default=False)
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-t', '--env_name', required=True, type=str)
    ap.add_argument('-t', '--env_type', required=True, choices=["atari", "vizdoom", "highway"], type=str)
    ap.add_argument('-t', '--task', required=True, choices=["train", "anim"], type=str)
    ap.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str)
    args = vars(ap.parse_args())

    if args['task'] == 'train':
        trainer = trainers.Trainer(args)
        trainer.train()
        
    elif args["task"] == "anim":
        assert args["load"] is not None, "Load a model for inference tasks"
        trainer = trainers.Trainer(args)
        trainer.create_animation()