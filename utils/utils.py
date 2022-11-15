import random
import os
import numpy as np
import torch
import argparse

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser(description='Training arguments for LD baseline model')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1, help='Fraction of data used as validation')
    parser.add_argument('--data', '-d', type=str, default='data/train/', help='Path to folder containing training data')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Model name from timm to be used as backbone')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--size', '-s', type=int, default=256, help='New image size')
    parser.add_argument('--seed', type=int, default=42, help='Set seed for reproducibility')

    return parser.parse_args()

class EarlyStopper:
    # Ref: https://stackoverflow.com/a/73704579
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta * self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False