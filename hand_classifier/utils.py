# coding=utf-8
import os
import matplotlib.pyplot as plt
import numpy as np

import conf

def img_show(img, text=None, save=False, color="white"):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                bbox={
                    'facecolor': color,
                    'alpha': 0.8,
                    'pad': 10
                }
        )
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

if __name__ == "__main__":
    pass
