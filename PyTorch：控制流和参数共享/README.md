# PyTorch：控制流和参数共享

作为动态图和权重共享的一个例子，我们实现了一个非常奇怪的模型：一个全连接的ReLU网络，在每一次前向传播时，它的隐藏层的层数为随机1到4之间的数，这样可以多次重用相同的权重来计算。

因为这个模型可以使用普通的Python流控制来实现循环，并且我们可以通过在定义转发时多次重用同一个模块来实现最内层之间的权重共享。

我们利用Mudule的子类很容易实现这个模型：


```python
# 可运行代码见本文件夹中的 dynamic_net.py
import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用。
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        对于模型的前向传播，我们随机选择0、1、2、3，
        并重用了多次计算隐藏层的middle_linear模块。
        由于每个前向传播构建一个动态计算图，
        我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
        在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
        这是Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 实例化上面定义的类来构造我们的模型
model = DynamicNet(D_in, H, D_out)

# 构造我们的损失函数（loss function）和优化器（Optimizer）。
# 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用了momentum方法。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    
    # 前向传播：通过向模型传入x计算预测的y。
    y_pred = model(x)

    # 计算并打印损失
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # 清零梯度，反向传播，更新权重 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
