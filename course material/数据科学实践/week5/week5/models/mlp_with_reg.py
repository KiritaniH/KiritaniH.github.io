import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO：先实现基础模型，随后添加BatchNorm或Dropout，观察测试loss变化情况
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # TODO 1: 定义第一个全连接层，将输入 (32 * 32 * 3) 映射到 512 维

        # TODO 2: 定义第二个全连接层，将 512 维映射到 1024 维

        # TODO 3: 定义第三个全连接层，将 1024 维映射到 512 维
        
        # TODO 4: 定义第四个全连接层，将 512 维映射到 10 维（分类任务）


    def forward(self, x):
        # TODO 5: 重新调整输入 x 的形状，使其变为 (batch_size, 32*32*3)
        # 提示：可以使用 view() 方法

        # TODO 6: 依次通过线性层和 ReLU 激活函数

        # TODO 7: 通过最后一层输出分类结果（不需要 ReLU）

        
        return x
    


    