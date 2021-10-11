
import os 
import trainers 
import argparse 
from datetime import datetime as dt


if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='Path to configuration file')
    ap.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str, help='Path to output directory')
    ap.add_argument('-r', '--resume', default=None, help='Path to output dir which contains saved state to resume training from')
    ap.add_argument('-l', '--load', default=None, help='Path to output dir from which pretrained models will be loaded')
    ap.add_argument('-t', '--task', default='train', type=str, help='Task to perform, choose from (train, resnet_train, viz, viz_1, gauss_viz)')
    args = vars(ap.parse_args())

    if args['task'] == 'train':
        trainer = trainers.Trainer(args)
        trainer.train()
