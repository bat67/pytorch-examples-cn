# 用例子学习PyTorch

> [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) 中文翻译版，部分意译，部分改动

> 翻译不对的地方拜托大家指出~

## 简介
这个repo通过自洽的示例介绍了PyTorch的基本概念。

PyTorch主要是提供了两个核心的功能特性：

* 一个类似于numpy的n维张量，但是可以在GPU上运行
* 搭建和训练神经网络时的自动微分/求导机制

我们将使用全连接的ReLU网络作为运行示例。该网络将有一个单一的隐藏层，并将使用梯度下降训练，通过最小化网络输出和真正结果的欧几里得距离，来拟合随机生成的数据。

## 环境

* PyTorch版本0.4及以上（PyTorch 1.0 **稳定版**已经发布，还有什么理由不更新呢~）

## 目录

### 张量(Tensors)

* [热身：使用NumPy](https://github.com/bat67/pytorch-examples-cn/tree/master/热身%EF%BC%9A使用NumPy)
* [PyTorch：张量(Tensors)](https://github.com/bat67/pytorch-examples-cn/tree/master/PyTorch%EF%BC%9A张量(Tensors))

### 自动求导(Autograd)

* [PyTorch：自动求导(Autograd)](https://github.com/bat67/pytorch-examples-cn/tree/master/PyTorch%EF%BC%9A自动求导(Autograd))
* [PyTorch：定义自己的自动求导函数（almost）](https://github.com/bat67/pytorch-examples-cn/tree/master/PyTorch%EF%BC%9A定义自己的自动求导函数)
* [TensorFlow：静态图](https://github.com/bat67/pytorch-examples-cn/tree/master/TensorFlow%EF%BC%9A静态图)

### `nn`模块(`nn` module)

* [PyTorch：神经网络模块nn](https://github.com/bat67/pytorch-examples-cn/tree/master/PyTorch%EF%BC%9A定制神经网络nn模块)
* [PyTorch：优化模块optim](https://github.com/bat67/pytorch-examples-cn/tree/master/PyTorch%EF%BC%9A优化模块optim)
* [PyTorch：定制神经网络nn模块（almost）](https://github.com/bat67/pytorch-examples-cn/tree/master/PyTorch%EF%BC%9A定制神经网络nn模块)
* [PyTorch：控制流和参数共享](https://github.com/bat67/pytorch-examples-cn/tree/master/PyTorch%EF%BC%9A控制流和参数共享)