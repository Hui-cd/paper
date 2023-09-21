# Pytorch Api
## torch
### 1. **torch.matmul**是用来对两个tensor的矩阵进行乘积
  - 如果tensor都是一维返回其点积(标量)
  - 如果两个参数都是二维的，则返回矩阵-矩阵乘积
  - 如果第一个参数是一维，第二个参数是二维，则为了矩阵乘法的目的，在其维度前添加 1。矩阵相乘后，前面的维度将被删除
  - 如果第一个参数是二维的，第二个参数是一维的，则返回矩阵向量乘积

例子：
``` python
# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
torch.matmul(tensor1, tensor2).size()
# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()
# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()
# batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
torch.matmul(tensor1, tensor2).size()
# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
torch.matmul(tensor1, tensor2).size() 
```

### 2. **torch.transpose(input, dim0, dim1) → Tensor** 
返回一个tensor其是input的转置，交换给点的维度dim0和dim1
### 3.**torch.t** 
其输入的维度小于等于2-d，返回input的转置

### 4. **Tensor.masked_fill_(mask, value)** 
在掩码为 True 时，用value填充自身张量的元素。掩码的形状必须与底层张量的形状一致。

Parameters:
- mask (BoolTensor) – the boolean mask
- value (float) – the value to fill in with

### 5.**functional.softmax(input, dim=None, _stacklevel=3, dtype=None)** 
它应用于沿 dim 的所有切片，并将重新缩放这些切片，使元素位于 [0, 1] 范围内，且总和为 1。

Parameters:
 - input (Tensor) – input
 - dim (int) – 计算 softmax 的维度。tensor(3,4), dim(-1)，表示最后一个，即tensor(3,4)中的4 进行softmax操作
 - dtype (torch.dtype, optional) – 返回张量的数据类型。如果指定，则在执行操作前将输入张量转换为 dtype。这有助于防止数据类型溢出。
 - Default: None.

### 6. **nn.ModuleList**
用于管理一组神经网络模块（nn.Module 的子类）。它类似于 Python 的列表（List），但具有一些附加功能，使其适用于神经网络模块的管理。下面是关于nn.ModuleList的详细解释：

#### 1. 初始化：
可以通过创建一个nn.ModuleList对象来初始化它，然后将各种nn.Module子类的实例添加到列表中

```python
import torch.nn as nn

module_list = nn.ModuleList([nn.Linear(10, 10), nn.ReLU(), nn.Dropout(0.5)])
```

#### 2.访问模块：

你可以像访问Python列表中的元素一样访问`nn.ModuleList`中的模块。例如，要访问列表中的第一个模块，可以使用索引：
``` python
first_module = module_list[0]
```
这将返回nn.Linear(10, 10)，即列表中的第一个模块。
#### 3.添加模块
可以使用append方法将新的模块添加到`nn.ModuleList`中：
```python
module_list.append(nn.BatchNorm2d(64))
```

#### 注意事项
  - 当使用nn.ModuleList时，PyTorch知道这些模块是模型的一部分，因此可以正确地跟踪模型的参数和梯度。
  - 如果你只是使用普通的Python列表来存储模块，PyTorch将不会识别它们作为模型的一部分，这可能会导致参数更新等问题。
#### 示例用途
nn.ModuleList通常用于以下情况：

  - 创建动态的、不定长度的神经网络模块。
  - 在自定义神经网络模型中，需要根据输入数据的特征数量动态地添加隐藏层。
  - 在循环神经网络（RNN）中，根据序列长度动态添加循环层。