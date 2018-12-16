# 可运行代码见本文件夹中的 two_layer_net_autograd.py
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# N是批大小；D_in是输入维度；
# H是隐藏层维度；D_out是输出维度  
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出数据
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# 产生随机权重tensor，将requires_grad设置为True意味着我们希望在反向传播时候计算这些值的梯度
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(500):

    # 前向传递:使用张量上的操作计算预测值y。由于w1和w2有requires_grad=True，涉及这些张量的操作将让PyTorch构建计算图，从而允许自动计算梯度。由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 计算并输出loss，loss是一个形状为()的张量，loss.item()是这个张量对应的python数值
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    
    # 使用autograd计算反向传播。这个调用将计算loss对所有requires_grad=True的tensor的梯度。这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()


    # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 反向传播之后手动置零梯度
        w1.grad.zero_()
        w2.grad.zero_()