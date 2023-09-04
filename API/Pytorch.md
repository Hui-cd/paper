# Pytorch Api
## torch
**torch.matmul**是用来对两个tensor的矩阵进行乘积
  - 如果tensor都是一维返回其点积(标量)
  - 如果两个参数都是二维的，则返回矩阵-矩阵乘积
  - 如果第一个参数是一维，第二个参数是二维，则为了矩阵乘法的目的，在其维度前添加 1。矩阵相乘后，前面的维度将被删除
  - 如果第一个参数是二维的，第二个参数是一维的，则返回矩阵向量乘积

例子：
``` 
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

