#coding=utf-8

import numpy
import matplotlib.pyplot as plt

def show_a_img(trainset):
    #show image demo
    (img, label) = trainset[2]
    
    show = tv.transforms.ToPILImage()
    img = img.numpy()    # FloatTensor转为ndarray
    img = numpy.transpose(img, (1,2,0)) #把channel那一维放到最后
    plt.imshow(img)
    plt.show()

