#coding=utf-8

import models
from config import DefaultConfig

opt = DefaultConfig()
lr = opt.lr
model = getattr(models, opt.model)
dataset = DogCat(opt.train_data_root)

def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    
    train_data = DogCat(opt.train_data_root, train=True)

def test():


def main():

if __name__ == '__main__':
    main()