import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size=32*32*3, num_classes=10):
        """
        Initialize the Linear Model for CIFAR-10 with a single linear layer, no non-linear activation.
        
        Args:
            input_size (int): Size of the flattened input (32x32x3 = 3072 for CIFAR-10)
            num_classes (int): Number of output classes (10 for CIFAR-10)
        """
        super(LinearModel, self).__init__()
        # TODO: 定义线性层
        # 1. 添加一个 nn.Flatten 层，将输入图像展平
        # 2. 添加一个 nn.Linear 层，将展平后的特征映射到 num_classes 个输出
        self.flatten =
        self.linear =

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10)
        """
        # TODO: 实现前向传播
        # 1. 将输入 x 展平
        # 2. 将展平后的结果传入线性层
        # 3. 返回输出
        x = 
        x = 
        return x

