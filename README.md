

# 1. siamese_network孪生网络

【参考】https://www.cnblogs.com/king-lps/p/8342452.html

+ siamess网络架构图
![IMAGE](imgs/siamess.png)

+ 损失函数：
![IMAGE](imgs/siamess_loss.png)

其中，m为容忍度， Dw为两张图片的欧氏距离：
![IMAGE](imgs/siamess_dw.png)

