

# 1. siamese_network孪生网络

【参考】https://www.cnblogs.com/king-lps/p/8342452.html

+ siamess网络架构图
![IMAGE](imgs/siamess.png)

+ 损失函数：
![IMAGE](imgs/siamess_loss.png)

其中，m为容忍度， Dw为两张图片的欧氏距离：
![IMAGE](imgs/siamess_dw.png)


+ 数据
    数据采用的是AT&T人脸数据。共40个人，每个人有10张脸。
    数据链接： https://files.cnblogs.com/files/king-lps/att_faces.zip
    首先解压后发现文件夹下共40个文件夹，每个文件夹里有10张pgm图片。这里生成一个包含图片路径的train.txt文件共后续调用：

-------------
# 2. 猫狗识别

+ 代码来源：《深度学习框架：pytorch入门与实践》--作者：陈云



--------------
# 3. 简单手势分类器

+ 代码来源：https://www.jianshu.com/p/c4723a4409cf


-----------------
# 4. triplet-loss网络

+ 代码来源：https://github.com/andreasveit/triplet-network-pytorch


----------------
# 5. word2vec



-----------------
# 6. 图片风格迁移










