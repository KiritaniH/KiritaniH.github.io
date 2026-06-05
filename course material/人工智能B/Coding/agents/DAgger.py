import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# ===================== 1. 基础模块（复用/适配你的项目） =====================
# 1.1 数据集类（兼容BC/DAgger）
class ExpertDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# 1.2 策略网络（与你现有DQN/BC网络结构对齐）
class PolicyNet(nn.Module):
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

    def predict(self, x):
        """推理时选最优动作"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1).item()

# 1.3 专家标注函数（CartPole规则专家，替代人工标注）
def expert_annotate(state):
    """专家为任意状态标注最优动作（CartPole规则：杆左偏左移，右偏右移）"""
    return 0 if state[2] < 0 else 1  # state[2]是杆的角度

# 1.4 基础训练/评估函数（复用）
def train_policy(model, dataloader, epochs=50, lr=1e-3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
    return loss_history

def evaluate_policy(env, model, num_episodes=20):
    model.eval()
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model.predict(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            done = terminated or truncated
        total_rewards.append(reward_sum)
    return np.mean(total_rewards), np.std(total_rewards)

# ===================== 2. DAgger核心实现 =====================
def dagger_algorithm(env, state_dim, action_dim, dagger_iterations=10, collect_episodes_per_iter=20, lr=1e-3):
    """
    DAgger主流程：
    - dagger_iterations: DAgger迭代轮数
    - collect_episodes_per_iter: 每轮收集的智能体轨迹数
    """
    # 步骤1：初始化数据集（纯专家数据）
    print("=== 初始化：生成纯专家数据集 ===")
    init_expert_states, init_expert_actions = [], []
    for _ in range(20):  # 初始专家数据量
        state, _ = env.reset()
        done = False
        while not done:
            action = expert_annotate(state)
            init_expert_states.append(state)
            init_expert_actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    agg_states = np.array(init_expert_states)
    agg_actions = np.array(init_expert_actions)

    # 步骤2：DAgger迭代
    model = PolicyNet(state_dim, action_dim)
    performance_history = []  # 记录每轮性能
    loss_history_per_iter = []  # 记录每轮训练损失

    for iter_idx in range(dagger_iterations):
        print(f"\n=== DAgger迭代 {iter_idx+1}/{dagger_iterations} ===")
        # 2.1 用当前数据集训练策略
        dataset = ExpertDataset(agg_states, agg_actions)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        loss_hist = train_policy(model, dataloader, lr=lr)
        loss_history_per_iter.append(loss_hist[-1])  # 记录最终损失

        # 2.2 评估当前策略性能
        mean_reward, std_reward = evaluate_policy(env, model)
        performance_history.append(mean_reward)
        print(f"迭代{iter_idx+1}性能：平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")

        # 2.3 用当前策略与环境交互，收集未标注的状态（智能体轨迹）
        new_states = []
        for _ in range(collect_episodes_per_iter):
            state, _ = env.reset()
            done = False
            while not done:
                new_states.append(state)  # 仅收集状态，不收集智能体动作
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = model.predict(state_tensor)  # 智能体选动作
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        # 2.4 专家为新状态标注最优动作
        new_actions = [expert_annotate(s) for s in new_states]
        new_states = np.array(new_states)
        new_actions = np.array(new_actions)

        # 2.5 聚合数据集（原数据 + 新标注数据）
        agg_states = np.concatenate([agg_states, new_states], axis=0)
        agg_actions = np.concatenate([agg_actions, new_actions], axis=0)
        print(f"聚合后数据集大小：{len(agg_states)} 条")

    # 步骤3：最终训练与评估
    print("\n=== DAgger迭代完成，最终训练 ===")
    final_dataset = ExpertDataset(agg_states, agg_actions)
    final_dataloader = DataLoader(final_dataset, batch_size=32, shuffle=True)
    train_policy(model, final_dataloader, lr=lr)
    final_mean_reward, final_std_reward = evaluate_policy(env, model)
    print(f"DAgger最终性能：平均奖励 = {final_mean_reward:.2f} ± {final_std_reward:.2f}")

    return model, performance_history, loss_history_per_iter

# ===================== 3. 对比实验（DAgger vs 纯BC） =====================
def compare_bc_vs_dagger(env, state_dim, action_dim):
    """对比纯BC和DAgger的效果"""
    # 3.1 纯BC实验（仅初始专家数据）
    print("===== 纯BC实验 =====")
    # 生成初始专家数据（与DAgger初始化一致）
    init_expert_states, init_expert_actions = [], []
    for _ in range(20):
        state, _ = env.reset()
        done = False
        while not done:
            action = expert_annotate(state)
            init_expert_states.append(state)
            init_expert_actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    bc_dataset = ExpertDataset(np.array(init_expert_states), np.array(init_expert_actions))
    bc_dataloader = DataLoader(bc_dataset, batch_size=32, shuffle=True)
    bc_model = PolicyNet(state_dim, action_dim)
    train_policy(bc_model, bc_dataloader)
    bc_mean_reward, bc_std_reward = evaluate_policy(env, bc_model)
    print(f"纯BC性能：平均奖励 = {bc_mean_reward:.2f} ± {bc_std_reward:.2f}")

    # 3.2 DAgger实验
    print("\n===== DAgger实验 =====")
    dagger_model, dagger_perf, dagger_loss = dagger_algorithm(env, state_dim, action_dim)

    # 3.3 结果可视化（保存到scores/目录，适配你的项目结构）
    plt.figure(figsize=(12, 5))
    # 性能对比
    plt.subplot(1, 2, 1)
    plt.bar(["纯BC", "DAgger"], [bc_mean_reward, dagger_perf[-1]], yerr=[bc_std_reward, 0], capsize=5)
    plt.title("BC vs DAgger 最终性能")
    plt.ylabel("平均奖励（CartPole满分500）")
    # DAgger迭代性能变化
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(dagger_perf)+1), dagger_perf, marker='o', label="DAgger每轮性能")
    plt.axhline(y=bc_mean_reward, color='r', linestyle='--', label="纯BC性能")
    plt.xlabel("DAgger迭代轮数")
    plt.ylabel("平均奖励")
    plt.legend()
    plt.tight_layout()
    # 保存到scores/目录（与你现有项目的score_logger.py输出路径一致）
    os.makedirs("scores", exist_ok=True)
    plt.savefig("scores/bc_vs_dagger.png")
    plt.show()

    return bc_model, dagger_model

# ===================== 4. 主函数（适配你的train.py入口） =====================
if __name__ == "__main__":
    # 初始化环境（CartPole-v1，与你现有DQN环境一致）
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 运行BC vs DAgger对比实验
    compare_bc_vs_dagger(env, state_dim, action_dim)

    # 关闭环境
    env.close()