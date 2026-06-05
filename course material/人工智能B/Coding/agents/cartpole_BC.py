import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===================== 1. 数据集生成模块 =====================
class ExpertDataset(Dataset):
    """自定义数据集类：加载状态-动作对"""
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)  # CartPole动作是离散的（0/1）

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def generate_expert_data(env, num_episodes=100, is_expert=True):
    """
    生成数据集：
    - is_expert=True：专家数据（手动规则/最优策略）
    - is_expert=False：随机策略数据（对比用）
    """
    states = []
    actions = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if is_expert:
                # 专家策略：CartPole简单规则（杆左偏则左移，右偏则右移）
                action = 0 if state[2] < 0 else 1  # state[2]是杆的角度
            else:
                # 随机策略
                action = env.action_space.sample()
            
            states.append(state)
            actions.append(action)
            
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(states), np.array(actions)

# ===================== 2. BC算法核心网络 =====================
class BCNet(nn.Module):
    """Behavioral Cloning网络：状态输入→动作分类"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# ===================== 3. 训练与评估模块 =====================
def train_bc(model, dataloader, optimizer, criterion, epochs=50):
    """训练BC模型"""
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for states, actions in dataloader:
            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(states)
        avg_loss = total_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return loss_history

def evaluate_policy(env, model, num_episodes=20):
    """评估训练后的策略"""
    model.eval()
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            done = terminated or truncated
        total_rewards.append(reward_sum)
    return np.mean(total_rewards), np.std(total_rewards)

# ===================== 4. 主流程：对比实验 =====================
if __name__ == "__main__":
    # 初始化环境
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ------------ 步骤1：生成两种数据集（专家 vs 随机）------------
    print("生成专家数据集...")
    expert_states, expert_actions = generate_expert_data(env, num_episodes=100, is_expert=True)
    print("生成随机数据集...")
    random_states, random_actions = generate_expert_data(env, num_episodes=100, is_expert=False)

    # ------------ 步骤2：构建数据加载器 ------------
    batch_size = 32
    expert_dataset = ExpertDataset(expert_states, expert_actions)
    expert_dataloader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
    
    random_dataset = ExpertDataset(random_states, random_actions)
    random_dataloader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True)

    # ------------ 步骤3：初始化模型与训练参数 ------------
    # 关键参数对比：学习率（1e-3 vs 1e-4）、隐藏层维度（64 vs 32）
    params_configs = [
        {"lr": 1e-3, "hidden_dim": 64, "name": "专家数据_ lr=1e-3_ hidden=64"},
        {"lr": 1e-4, "hidden_dim": 64, "name": "专家数据_ lr=1e-4_ hidden=64"},
        {"lr": 1e-3, "hidden_dim": 32, "name": "专家数据_ lr=1e-3_ hidden=32"},
        {"lr": 1e-3, "hidden_dim": 64, "name": "随机数据_ lr=1e-3_ hidden=64"},  # 对比数据集
    ]

    # 存储结果
    results = {}
    loss_histories = {}

    # 遍历所有参数配置训练
    for cfg in params_configs:
        print(f"\n===== 训练配置：{cfg['name']} =====")
        model = BCNet(state_dim, action_dim, hidden_dim=cfg["hidden_dim"])
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.CrossEntropyLoss()

        # 选择数据集
        dataloader = expert_dataloader if "专家" in cfg["name"] else random_dataloader

        # 训练
        loss_hist = train_bc(model, dataloader, optimizer, criterion, epochs=50)
        loss_histories[cfg["name"]] = loss_hist

        # 评估
        mean_reward, std_reward = evaluate_policy(env, model)
        results[cfg["name"]] = (mean_reward, std_reward)
        print(f"评估结果：平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")

    # ------------ 步骤4：结果可视化 ------------
    # 1. 奖励对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    names = list(results.keys())
    rewards = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    plt.bar(names, rewards, yerr=stds, capsize=5)
    plt.title("不同配置下的策略平均奖励")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("平均奖励")

    # 2. 损失曲线对比（选前3个配置）
    plt.subplot(1, 2, 2)
    for name, loss in loss_histories.items():
        plt.plot(loss, label=name)
    plt.title("训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 关闭环境
    env.close()