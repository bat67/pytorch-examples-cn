# PyTorch：定义自己的自动求导函数

在底层，每一个原始的自动求导运算实际上是两个在Tensor上运行的函数。其中，**forward**函数计算从输入Tensors获得的输出Tensors。而**backward**函数接收输出Tensors对于某个标量值的梯度，并且计算输入Tensors相对于该相同标量值的梯度。 

在PyTorch中，我们可以很容易地通过定义`torch.autograd.Function`的子类并实现`forward`和`backward`函数，来定义自己的自动求导运算。之后我们就可以使用这个新的自动梯度运算符了。然后，我们可以通过构造一个实例并像调用函数一样，传入包含输入数据的tensor调用它，这样来使用新的自动求导运算。

这个例子中，我们自定义一个自动求导函数来展示ReLU的非线性。并用它实现我们的两层网络：

```python
# 可运行代码见本文件夹中的 two_layer_net_custom_function.py
import torch

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass we receive a context object and a Tensor containing the
        input; we must return a Tensor containing the output, and we can use the
        context object to cache objects for use in the backward pass.
        """
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the context object and a Tensor containing
        the gradient of the loss with respect to the output produced during the
        forward pass. We can retrieve cached data from the context object, and must
        compute and return the gradient of the loss with respect to the input to the
        forward function.
        """
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x


device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and output
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; we call our
    # custom ReLU implementation using the MyReLU.apply function
    y_pred = MyReLU.apply(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    with torch.no_grad():
        # Update weights using gradient descent
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()

```