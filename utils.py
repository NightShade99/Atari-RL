
import os
import torch
import logging
import numpy as np
from datetime import datetime

__all__ = ['pbar', 'AvgMeter', 'Logger']

COLORS = {
    "yellow": "\x1b[33m", 
    "blue": "\x1b[94m", 
    "green": "\x1b[32m",
    "red": "\x1b[33m", 
    "end": "\033[0m"
}


def pbar(p=0, msg="", bar_len=20):
    msg = msg.ljust(50)
    block = int(round(bar_len * p))
    text = "\rProgress: [{}] {}% {}".format(
        "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
        round(p * 100, 2),
        msg,
    )
    print(text, end="" if p < 1 else "\n")


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        for key, value in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = [value] 
            else:
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])
    
    
class Logger:

    def __init__(self, output_dir):
        [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
        logging.basicConfig(
            level = logging.INFO,
            format = "%(message)s",
            handlers = [logging.FileHandler(os.path.join(output_dir, "trainlogs.txt"))])

    def print(self, msg, mode=""):
        if mode == "info":
            print(f"{COLORS['yellow']}[INFO] {msg}{COLORS['end']}")
        elif mode == 'train':
            print(f"[TRAIN] {msg}")
        elif mode == 'val':
            print(f"{COLORS['blue']}[VALID] {msg}{COLORS['end']}")
        else:
            print(f"{msg}")

    def write(self, msg, mode=''):
        if mode == "info":
            msg = f"[INFO] {msg}"
        elif mode == "train":
            msg = f"[TRAIN] {msg}"
        elif mode == "val":
            msg = f"[VALID] {msg}"
        logging.info(msg)

    def record(self, msg, mode):
        self.print(msg, mode)
        self.write(msg, mode)
        
    def record_args(self, args):
        self.record("\n---- experiment configuration ----", mode='')
        args_ = vars(args)
        for arg, value in args_.items():
            self.record(f" * {arg} => {value}", mode='')
        self.record("----------------------------------", mode='')