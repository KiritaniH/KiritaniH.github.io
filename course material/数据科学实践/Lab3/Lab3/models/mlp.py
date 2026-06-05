import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__() 
        # TODO: 定义层
        # 1. 定义第一个全连接层：输入 32*32*3 (3072)，输出 64
        # 2. 定义第二个全连接层：输入 64，输出 128
        # 3. 定义第三个全连接层：输入 128，输出 128
        # 4. 定义第四个全连接层：输入 128，输出 10 (类别数)
        
        self.fc1 = 
        self.fc2 =
        self.fc3 = 
        self.fc4 = 
     
    def forward(self,x):
        # TODO: 实现前向传播
        # 1. 将输入 x 展平为 (batch_size, 3072)
        # 2. 通过第一个全连接层并应用 ReLU 激活
        # 3. 通过第二个全连接层并应用 ReLU 激活
        # 4. 通过第三个全连接层并应用 ReLU 激活
        # 5. 通过第四个全连接层（不加激活），返回输出

        return x

    