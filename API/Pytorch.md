# Pytorch Api

## torch

### torch.matmul

是用来对两个tensor的矩阵进行乘积

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

### torch.transpose(input, dim0, dim1) → Tensor

返回一个tensor其是input的转置，交换给点的维度dim0和dim1

### torch.t

其输入的维度小于等于2-d，返回input的转置

### Tensor.masked_fill_(mask, value)

在掩码为 True 时，用value填充自身张量的元素。掩码的形状必须与底层张量的形状一致。

Parameters:

- mask (BoolTensor) – the boolean mask
- value (float) – the value to fill in with

### functional.softmax(input, dim=None, _stacklevel=3, dtype=None)

它应用于沿 dim 的所有切片，并将重新缩放这些切片，使元素位于 [0, 1] 范围内，且总和为 1。

Parameters:

- input (Tensor) – input
- dim (int) – 计算 softmax 的维度。tensor(3,4), dim(-1)，表示最后一个，即tensor(3,4)中的4 进行softmax操作
- dtype (torch.dtype, optional) – 返回张量的数据类型。如果指定，则在执行操作前将输入张量转换为 dtype。这有助于防止数据类型溢出。
- Default: None.

### **nn.ModuleList**

用于管理一组神经网络模块（nn.Module 的子类）。它类似于 Python 的列表（List），但具有一些附加功能，使其适用于神经网络模块的管理。下面是关于nn.ModuleList的详细解释：

#### 初始化

可以通过创建一个nn.ModuleList对象来初始化它，然后将各种nn.Module子类的实例添加到列表中

```python
import torch.nn as nn

module_list = nn.ModuleList([nn.Linear(10, 10), nn.ReLU(), nn.Dropout(0.5)])
```

#### 访问模块

你可以像访问Python列表中的元素一样访问`nn.ModuleList`中的模块。例如，要访问列表中的第一个模块，可以使用索引：

``` python
first_module = module_list[0]
```

这将返回nn.Linear(10, 10)，即列表中的第一个模块。

#### 添加模块

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

### copy.deepcopy(module)

这部分使用 Python 的 copy 模块中的 deepcopy 函数来创建一个模块的深度复制。深度复制是一种复制对象的方法，它会递归地复制对象及其所有嵌套对象，以确保原始对象和复制后的对象之间没有共享状态。在这里，module 是要复制的神经网络模块。

### view()

PyTorch 中的一个用于张量操作的重要方法，它用于改变张量的形状，但并不改变张量的数据

```python
new_tensor = tensor.view(*shape)
```

- **tensor**: 这是要操作的原始张量。
- **shape**: 这是一个可变参数，表示你想要将原始张量变换成的新形状。新形状的每个维度大小必须与原始张量的总元素数量一致，或者某些维度的大小可以使用 -1 自动推断。
- **view** 方法的主要目的是改变张量的形状，而不需要复制或重新分配存储空间，因此它通常非常高效。它非常适用于需要改变张量形状以适应不同操作的情况，例如在神经网络中进行不同层之间的数据传递。#
  
#### view 方法的一些示例

- 简单重塑:

``` python
      # 创建一个4x4的张量
      x = torch.arange(16).view(4, 4)
```

这会将原始的1维张量重塑为一个4x4的二维张量。

- 自动推断大小:

``` python
      # 创建一个1x8的张量，自动推断第二个维度为4
      x = torch.arange(8).view(1, -1)
```

在这个例子中，-1 用于自动推断第二个维度的大小，以确保总元素数量保持不变。

- 与其他操作组合：

``` python
      # 将一个3x3x3的张量重塑为一个27维的1维张量
      x = torch.arange(27).view(-1)
```

这个示例将一个3维张量重塑为一个1维张量，但总元素数量保持不变。

- 注意事项：
  - 使用 view 时，要确保新形状的各维度大小与原始张量的总元素数量一致，否则会引发错误。
  - view 操作不会复制张量的数据，因此新张量和原始张量共享相同的数据存储，修改一个张量会影响到另一个张量。

### contiguous()

是 PyTorch 中用于处理张量内存连续性的方法。它通常用于解决一些需要连续内存的操作或问题

```python
import torch

# 创建一个非连续的张量
x = torch.tensor([[1, 2], [3, 4]])
y = x[:, 1]  # 从 x 中选择一列，y 不是连续的

# 尝试对非连续张量进行某些操作可能会引发错误
# 例如，尝试进行 reshape 操作
# z = y.view(2, 1)  # 这会引发错误

# 使用 .contiguous() 方法确保连续性
y = y.contiguous()

# 现在可以进行 reshape 操作
z = y.view(2, 1)

```

在这个示例中，y 最初不是连续的，因此尝试对其进行 view 操作会引发错误。通过使用 contiguous() 方法，我们确保了 y 是连续的，从而使 view 操作成为可能。

contiguous() 的一些常见应用场景包括：在进行视图操作（如 view、reshape）之前，确保张量是连续的；在进行某些需要连续内存的操作时，如使用 NumPy 操作或与其他库进行交互。但需要注意，使用 contiguous() 会创建一个新的张量，这可能会导致内存复制，因此在性能敏感的情况下要小心使用。

### torch.nn.Dropout(p=0.5, inplace=False)

 是 PyTorch 中用于实现 dropout 正则化的模块之一。Dropout 是一种正则化技术，用于减少神经网络的过拟合现象，从而提高模型的泛化能力。在训练过程中，Dropout 随机地将一部分神经元的输出置零，这样每个神经元都不能依赖于其他特定神经元的输出，从而迫使网络更加鲁棒。

#### torch.nn.Dropout 接受以下参数

- p（概率）：这是一个介于0和1之间的浮点数，表示要丢弃的神经元的概率。具体来说，它表示每个神经元在前向传播期间被置零的概率。例如，如果 p=0.5，那么每个神经元在前向传播期间有50%的概率被置零。这个参数是必需的。

- inplace（布尔值）：如果设置为 True，则会在原地修改输入张量，即直接将输入张量的某些元素置零，而不复制张量。默认情况下，它是 False，表示不会修改原始输入张量，而是返回一个新的张量，其中某些元素被置零。

 使用示例：

 ```python
  import torch
  import torch.nn as nn

  # 创建一个Dropout层，丢弃概率为0.5，不在原地修改输入
  dropout_layer = nn.Dropout(p=0.5, inplace=False)

  # 输入张量
  x = torch.randn(3, 3)

  # 应用Dropout层
  y = dropout_layer(x)

  # y是一个新的张量，x不变

 ```

 ``` python
 # 创建一个Dropout层，丢弃概率为0.5，在原地修改输入
dropout_layer = nn.Dropout(p=0.5, inplace=True)

# 输入张量
x = torch.randn(3, 3)

# 应用Dropout层
y = dropout_layer(x)

# y和x是同一个张量，x被修改了

 ```

 在训练深度神经网络时，通常会在模型的隐藏层之间添加 Dropout 层，以减少过拟合风险，并提高模型的泛化性能。但在推理阶段，通常不需要使用 Dropout。

### torch.nn.parameter.Parameter

它用于将一个张量标记为模型的可学习参数（learnable parameter）。这个类通常用于定义神经网络的权重（weights）和偏置（biases），以便在训练过程中通过反向传播来更新它们。

Parameter 类是 torch.Tensor 的子类，它继承了所有张量的操作和性质，但具有一个额外的属性 requires_grad，它用于指示是否需要对这个参数进行梯度计算，以便进行反向传播更新参数。

#### 参数及其详细解释

- data（张量，可选）：这是一个张量，即参数的初始值。如果未提供此参数，则参数的初始值将是随机的或根据模型的初始化策略确定的。通常情况下，你会提供一个初始的张量值。默认值是 None。

- requires_grad（布尔值，可选）：指示是否需要计算这个参数的梯度。如果设置为 True，则该参数将在反向传播中被跟踪并计算梯度。如果设置为 False，则不会计算梯度，这对于一些不需要训练的参数（如固定的嵌入层）非常有用。默认值是 True。

``` python
import torch
import torch.nn.parameter as Parameter

# 创建一个形状为 (3, 4) 的可学习参数，初始值随机生成
param = Parameter(torch.randn(3, 4))

# 创建一个形状为 (2, 2) 的可学习参数，初始值为指定的张量值
initial_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
param2 = Parameter(initial_data)

# 创建一个不需要计算梯度的可学习参数
param_no_grad = Parameter(torch.zeros(2, 2), requires_grad=False)

```

通常，在定义神经网络模型时，你会将可学习参数作为 Parameter 对象添加到模型中，以便在训练中通过优化器来更新这些参数的值。这样，PyTorch 将自动跟踪这些参数的梯度，以便进行反向传播和优化。
