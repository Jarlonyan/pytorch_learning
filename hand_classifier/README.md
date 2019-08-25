


数据下载链接：https://cloud.tsinghua.edu.cn/f/787490e187714336aae2/?dl=1


1. 不要以为加了下面的代码，base_model就不会训练了，参数仍在更新

```
for param in self.emb_model.parameters():
    param.requires_grad = False
```


需要自己做一下过滤
```
optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

