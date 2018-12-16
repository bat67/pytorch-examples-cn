# PyTorch：张量(Tensors)

NumPy是一个很棒的框架，但是它不支持GPU以加速运算。现代深度神经网络，GPU常常提供[50倍以上的加速]((https://github.com/jcjohnson/cnn-benchmarks))，所以NumPy不能满足当代深度学习的需求。 

我们先介绍PyTorch最基础的概念：**张量（Tensor）**。逻辑上，PyTorch的tensor和NumPy array是一样的：tensor是一个n维数组，PyTorch提供了很多函数操作这些tensor。任何希望使用NumPy执行的计算也可以使用PyTorch的tensor来完成；可以认为它们是科学计算的通用工具。

和NumPy不同的是，PyTorch可以利用GPU加速。要在GPU上运行PyTorch张量，在构造张量使用`device`参数把tensor建立在GPU上。

这里我们利用PyTorch的tensor在随机数据上训练一个两层的网络。和前面NumPy的例子类似，我们使用PyTorch的tensor，手动在网络中实现前向传播和反向传播： 


```python
# 可运行代码见本文件夹中的 two_layer_net_tensor.py
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# N是批大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出数据
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：计算预测值y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算并输出loss；loss是存储在PyTorch的tensor中的标量，维度是()（零维标量）；我们使用loss.item()得到tensor中的纯python数值。
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 反向传播，计算w1、w2对loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```
