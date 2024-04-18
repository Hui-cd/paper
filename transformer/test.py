import torch
import torch.nn.parameter as Parameter

import torch
import torch.nn as nn

import torch
import torch.nn as nn

# 假设x是从卷积层得到的输出，形状为 [batch_size, 32, 32, 10]
x = torch.randn(1, 32, 32, 10)  # 模拟一批数据中的一个样本

# 展平操作
x_flattened = x.view(x.size(0), -1)  # 变换成 [1, 10240]

# 全连接层
fc_layer = nn.Linear(10240, 10)  # 从10240维降到128维
output = fc_layer(x_flattened)  # 得到全连接层的输出

# 激活函数
activated_output = nn.ReLU()(output)

# 输出结果
print("Output of the fully connected layer:", activated_output)
