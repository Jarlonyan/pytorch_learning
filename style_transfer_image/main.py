#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import style_transfer_model


'''
实现图像的风格迁移
代码来源：《深度学习框架:PyTorch入门与实践》--作者：陈云
'''

def main():
   transform = transform.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ]
    )
    

if __name__ == '__main__':
    main()
