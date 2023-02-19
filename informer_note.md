# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
## What is Time Series Forecasting?
时间序列预测(time series forcasting) 任务在工业中有着广泛的应用，例如预测股票价格、预测气象数据、预测销售数据等等。时间序列预测任务的目标是预测未来的时间序列值。时间序列预测任务的输入是一个长度为 $T$ 的时间序列 $\mathbf{x} = (x_1, x_2, \cdots, x_T)$，输出是一个长度为 $H$ 的时间序列 $\mathbf{y} = (y_1, y_2, \cdots, y_H)$，其中 $H$ 是预测的时间步数。时间序列预测任务的目标是预测未来的 $H$ 个时间步的值。时间序列预测任务的输入和输出都是一维的，因此时间序列预测任务也被称为一维时间序列预测任务。

而Long time series forecasting任务是指输入的时间序列长度 $T$ 很长，例如 $T=10000$。Long time series forecasting任务的目标是预测未来的 $H$ 个时间步的值。Long time series forecasting任务的输入和输出都是一维的，因此Long time series forecasting任务也被称为一维Long time series forecasting任务。

LSTF 由于预测序列长，所以模型具有较强的长距离依赖性，因此需要使用较大的模型来学习长距离依赖性。但是，较大的模型会导致计算量大，训练时间长，因此需要使用较大的计算资源来训练模型。因此，LSTF 任务的研究方向是设计高效的模型来学习长距离依赖性。

## Transformer 的问题：
* self-attention 的时间和空间复杂度是O(N^2)，因此在长序列上计算复杂度很高。因此，Transformer 不能很好地解决长序列预测任务。
* encoder-decoder结构在解码时step-by-step,因此在预测长序列时，需要预测的时间步数越多，计算复杂度越高。

针对以上问题，Informer 在 Transformer 的基础上提出了三个改进：
* 1.使用了ProbSparse self-attention，将时间复杂度从O(N^2)降低到O(NlogN)。
* 2.提出了self-attention的蒸馏机制，将空间复杂度从O(N^2)降低到O(N)。
* 3.提出了生成式的解码机制，将解码的时间复杂度从O(N)降低到O(1)。


