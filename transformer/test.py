import torch
import torch.nn.parameter as Parameter

# 创建一个形状为 (3, 4) 的可学习参数，初始值随机生成
param = Parameter(torch.randn(3, 4))

# 创建一个形状为 (2, 2) 的可学习参数，初始值为指定的张量值
initial_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
param2 = Parameter(initial_data)

# 创建一个不需要计算梯度的可学习参数
param_no_grad = Parameter(torch.zeros(2, 2), requires_grad=False)