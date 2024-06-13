An official source code for paper Deep Incomplete Multi-view Clustering with Cross-view Partial Sample and Prototype Alignment, accepted by CVPR 2023.


```linux
|---Nmetrics.py        总evaluate函数，返回acc,nmi,purity,fscore,precision,recall,ari几个指标
|---alignment.py       对齐模块、对应文章3.3 那一章节
|---loss.py            计算了原型对齐和实例对齐的损失(Instance_loss和Prototype_loss)
|---main.py            main.py
|---network.py         定义了一个多视图自编码器网络结构，用于从多个视角输入的数据中提取特征
|---utils.py           clustering函数、欧氏距离的计算、余弦相似度的计算等通用函数
```
