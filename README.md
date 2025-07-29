# AISkater

**AISkater** 旨在实现滑动输入识别

## v0.9

实现了由滑动序列 (x, y, dx, dy, r, dt) 到字母序列的生成
总体架构为Encoder-Decoder结构，配有attention模块，Encoder和Decoder均由两层GRU构成  
完成了模型搭建和数据生成，进行了cpu上随机数据的梯度下降训练过程验证，150个数据点用时24min  
随后，将生成数据接入模型，出现若干问题

sft at 25.7.23

## v1

完成这一版模型，实现生成数据下的训练。于cpu上运行成功

TODO
1. 实现eval (done at 7.25)
2. 修改模型，实现产生真实单词 (done at **v2**)

sft at 25.7.24

实现eval

sft at 25.7.25

## v2

这一版模型旨在对序列和单词分别嵌入后对齐

完成两个嵌入模型并运行训练

TODO
1. 实现对两个嵌入模型的eval
2. 完成对齐模型 (done at 7.29)

sft at 25.7.28

完成对齐模型并运行训练
实现部署，在puma.txt下构建单词表完成测试
准确率极低

TODO
1. 调整模型，解决嵌入空间崩溃问题

sft at 25.7.29