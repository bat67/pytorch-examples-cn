# PyTorch：神经网络模块nn


计算图和autograd是十分强大的工具，可以定义复杂的操作并自动求导；然而对于大规模的网络，autograd太过于底层。

在构建神经网络时，我们经常考虑将计算安排成**层**，其中一些具有**可学习的参数**，它们将在学习过程中进行优化。

TensorFlow里，有类似[Keras](https://github.com/fchollet/keras)，[TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)和[TFLearn](http://tflearn.org/)这种封装了底层计算图的高度抽象的接口，这使得构建网络十分方便。 

在PyTorch中，包`nn`完成了同样的功能。`nn`包中定义一组大致等价于层的**模块**。一个模块接受输入的tesnor，计算输出的tensor，而且还保存了一些内部状态比如需要学习的tensor的参数等。`nn`包中也定义了一组损失函数（loss functions），用来训练神经网络。 

这个例子中，我们用`nn`包实现两层的网络：


```python
# 可运行代码见本文件夹中的 two_layer_net_nn.py
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出随机张量
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)


# 使用nn包将我们的模型定义为一系列的层。
# nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出。
# 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量。
# 在构造模型之后，我们使用.to()方法将其移动到所需的设备。
model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        ).to(device)


# nn包还包含常用的损失函数的定义；
# 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数。
# 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值;
# 这是为了与前面我们手工计算损失的例子保持一致，
# 但是在实践中，通过设置reduction='elementwise_mean'来使用均方误差作为损失更为常见。
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):

    # 前向传播：通过向模型传入x计算预测的y。
    # 模块对象重载了__call__运算符，所以可以像函数那样调用它们。
    # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量。
    y_pred = model(x)
    
    # 计算并打印损失。我们传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量。
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    
    # 反向传播之前清零梯度
    model.zero_grad()

    # 反向传播：计算模型的损失对所有可学习参数的导数（梯度）。
    # 在内部，每个模块的参数存储在requires_grad=True的张量中，
    # 因此这个调用将计算模型中所有可学习参数的梯度。
    loss.backward()

    # 使用梯度下降更新权重。
    # 每个参数都是张量，所以我们可以像我们以前那样可以得到它的数值和梯度
    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad
```