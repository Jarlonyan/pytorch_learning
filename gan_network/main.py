# coding=utf-8

import argparse
import torch
import torchvision
import torchvision.utils vutils
import torch.nn as nn
from random import randint

from model import NetD,NetG
import conf

def main():
    train()
    #test()

if __name__ == "__main__":
    main()
