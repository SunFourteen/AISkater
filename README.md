# AISkater

**AISkater**旨在实现滑动输入识别

## v0.9

实现了由滑动序列$(x, y, dx, dy, r, dt)$到字母序列的生成
总体架构为Encoder-Decoder结构，配有attention模块，Encoder和Decoder均由两层GRU构成  
完成了模型搭建和数据生成，进行了cpu上随机数据的梯度下降训练过程验证，150个数据点用时24min  
随后，将生成数据接入模型，出现若干问题

sft at 25.7.23

## v1

完成这一版模型，实现生成数据下的训练。于cpu上运行成功

dataset: puma.txt  
data size: 260  
max length: 167  
epoch: 20  
batch size: 16  
loss: 0.7360  
run time: 45min

TODO
1. 实现eval
2. 修改模型，实现产生真实单词

sft at 25.7.24

实现eval

sft at 25.7.25