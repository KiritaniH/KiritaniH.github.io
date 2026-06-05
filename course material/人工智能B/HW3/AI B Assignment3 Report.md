# AI B Assignment3 Report

#### 12312515 王洛源

### 一.  实验原理

强化学习（Reinforcement Learning, RL）是一种让智能体通过与环境交互获得经验、并基于奖惩机制改进行为的机器学习方法。智能体在每一步根据当前状态 $s_t$ 选择一个动作 $a_t$​，环境返回奖励 $r_{t+1}$​ 和下一个状态 $s_{t+1}$。其核心目标是通过不断试错学习出一个最大化累计回报的最优策略 $π^*$。

在本次作业中，我们主要研究两种基于值函数的学习方法：**TD-learning** 和 **Q-learning**。它们都是通过采样得到经验序列，然后不断更新估计值，从而逐渐逼近真实的状态价值或动作价值。

##### （1）TD-learning 原理

TD-learning 是一种结合了动态规划和蒙特卡洛思想的预测算法。它在每一次与环境交互后，用当前的估计值来修正对状态的价值判断，而不需要等待整个回合结束。其更新规则如下：

$$
V(s)←V(s)+\alpha (R(s,\pi (s),s')+\gamma V(s')-V(s))
$$

其中，

- **α** 为学习率（控制每次更新幅度），

- **γ** 为折扣因子（衡量对未来奖励的重视程度）。

通过多次交互和参数更新，智能体能够逐步逼近在当前策略下的状态价值函数 $V^π​(s)$。

##### （2）Q-learning 原理

Q-learning 是一种无模型的强化学习控制算法，直接学习 $Q^*(s,a)$，不依赖环境的转移概率模型。其更新规则为：

$$
Q(s, a) ← (1 - α) · Q(s, a) + α · [ r + γ · maxₐ′ Q(s′, a′) ]
$$

通过不断重复该更新，$Q(s, a)$ 会逐步逼近 $Q^*(s, a)$，从而学习到最优策略：

$$
π^*(s) = argmaxₐ Q(s, a)
$$

### 二.  实验设计和过程

代码实现见code文件夹，其中TD-learning对学习策略做出了修改，改为贪心策略方便快速收敛（注释中注明可修改的部分）。下面展示对以上两种学习方法，应用不同参数指令学习后得到的结果

##### （1）TD-learning部分

首先运行作业文档中给出的示例代码：

```
py main.py -a td -g BookGrid -k 200 --alpha 0.5 --discount 0.9
```

结果如下：其中第一张图是状态价值函数收敛图，第二张图是运行结束后终端和可视化窗口截图（下同）

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/ut1.png" title="" alt="" width="389">     <img src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/result1.png" title="" alt="" width="282">

可以看到学习还不够稳定，平均回报还是较低，因此我们增加回合数，降低学习率：

```
py main.py -a td -g BookGrid -k 1000 --alpha 0.3 --discount 0.9
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/ut2.png" title="" alt="" width="389">     <img title="" src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/result2.png" alt="" width="264">

平均回报有所上涨，下一步我们进一步降低学习率：

```
py main.py -a td -g BookGrid -k 1000 --alpha 0.1 --discount 0.9
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/ut3.png" title="" alt="" width="389">     <img title="" src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/result3.png" alt="" width="280">

函数收敛图呈现稳定趋势，下一步我们略微下降$\gamma $的值：

```
py main.py -a td -g BookGrid -k 1000 --alpha 0.1 --discount 0.85
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/ut4.png" title="" alt="" width="390">     <img title="" src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/result4.png" alt="" width="279">

折扣因子γ下降后，模型对未来奖励的重视程度减弱，导致总体回报下降，但收敛更稳定。

最后我们保持此前0.1的学习率，进一步增加回合数：

```
py main.py -a td -g BookGrid -k 3000 --alpha 0.1 --discount 0.9
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/ut5.png" title="" alt="" width="390">     <img title="" src="file:///C:/Users/lenovo/Desktop/AIb/code/TD%20Figs/result5.png" alt="" width="266">

##### （2）Q-learning部分

首先尝试运行下面的指令：

```
py main.py -a q -g BookGrid -k 500 --alpha 0.5 --epsilon 0.1 --discount 0.9
```

学习曲线图结果如下：

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/QL%20Figs/LC1.png" title="" alt="" width="391"> 

运行下面代码进行调参：

```
py main.py -a q -g BookGrid -k 1500 --alpha 0.10 --epsilon 0.05 --discount 0.95
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/QL%20Figs/LC2.png" title="" alt="" width="396">

在该组参数下，模型收敛较快且波动较小，后续参数调整均以此为基准。

```
py main.py -a q -g BookGrid -k 1500 --alpha 0.10 --epsilon 0.05 --discount 0.85
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/QL%20Figs/LC3.png" title="" alt="" width="405">

```
py main.py -a q -g BookGrid -k 1500 --alpha 0.10 --epsilon 0.10 --discount 0.95
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/QL%20Figs/LC4.png" title="" alt="" width="407">

```
py main.py -a q -g BookGrid -k 1500 --alpha 0.30 --epsilon 0.05 --discount 0.95
```

<img src="file:///C:/Users/lenovo/Desktop/AIb/code/QL%20Figs/LC5.png" title="" alt="" width="413">

### 三. 实验总结和参数分析

从整体结果来看，**TD-learning** 与 **Q-learning** 都能够在一定回合数后收敛到较优的策略，但二者的参数敏感性存在明显差异。

#### (1) TD-learning 参数特征

- **学习率 α**：在较高学习率（如 0.5）下，更新幅度过大，导致波动显著；逐步降低到 0.1 后，曲线收敛平稳且平均回报上升，说明较小 α 更有助于在噪声环境中稳定学习。  
- **折扣因子 γ**：从 0.9 调低至 0.85 后，模型对长期奖励的关注度下降，导致平均回报从约 0.5 降至 0.2 左右，但曲线更加平滑，体现出“稳定与收益”的权衡关系。  
- **回合数 k**：延长训练回合数（如从 1000 增至 3000）能有效减少早期波动，使状态价值函数逐渐趋于稳定。

#### (2) Q-learning 参数特征

- **学习率 α**：中等学习率（0.10 左右）表现最佳；过大（如 0.30）会导致收敛不稳，过小则学习过慢。  
- **探索率 ε**：较小 ε（0.05）有利于在后期集中利用已有知识、提高回报；当 ε=0.10 时，曲线抖动略增，反映出探索动作带来的波动性。  
- **折扣因子 γ**：γ=0.95 时平均回报最高，γ=0.85 时学习更稳但回报下降，说明适度强调未来奖励能提升策略质量。  
- **训练回合 k**：在 1500 回合左右已能获得较好稳定性，继续增加回合数提升有限。

#### (3) 综合分析

TD-learning 在本任务中收敛速度较慢但稳定性较高，适合策略评估类任务；而 Q-learning 能更快提升平均回报，但对参数更敏感。总体而言，  

- **较小 α、适中 γ（≈0.9~0.95）及低 ε（≈0.05）** 是平衡收敛与收益的最佳组合。  
- 从曲线走势看，Q-learning 的移动平均线在 300~400 回合后趋于稳定，说明其在有限训练内即可获得接近最优的策略。

综上，本实验验证了不同超参数对强化学习算法性能的影响规律，为在更复杂环境下的参数选择提供了经验参考。
