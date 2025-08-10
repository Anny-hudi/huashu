# 第五题解决方案 - 能耗优化

# 导入必要的库
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 系统参数
NUM_USERS = 70  # U1-U10 (URLLC), e1-e20 (eMBB), m1-m40 (mMTC)
NUM_BS = 4      # MBS_1, SBS_1, SBS_2, SBS_3
NUM_SBS = 3     # SBS数量
MBS_RESOURCE_BLOCKS = 100  # 宏基站资源块数
SBS_RESOURCE_BLOCKS = 50   # 微基站资源块数

# 基站位置
BS_POSITIONS = {
    0: (0, 0),           # MBS_1
    1: (0, 500),         # SBS_1
    2: (-433.0127, -250), # SBS_2
    3: (433.0127, -250)   # SBS_3
}

# 用户类型映射
USER_TYPES = {}
for i in range(10):  # U1-U10: URLLC
    USER_TYPES[i] = 'URLLC'
for i in range(10, 30):  # e1-e20: eMBB
    USER_TYPES[i] = 'eMBB'
for i in range(30, 70):  # m1-m40: mMTC
    USER_TYPES[i] = 'mMTC'

# 用户索引到列名的映射
USER_COLUMN_MAP = {}
for i in range(10):  # U1-U10: URLLC
    USER_COLUMN_MAP[i] = f'U{i+1}'
for i in range(10, 30):  # e1-e20: eMBB
    USER_COLUMN_MAP[i] = f'e{i-9}'
for i in range(30, 70):  # m1-m40: mMTC
    USER_COLUMN_MAP[i] = f'm{i-29}'

# 用户接入模式映射
ACCESS_MODES = {
    0: 'Direct',  # 直接接入
    1: 'Relay',   # 中继接入
    2: 'D2D'      # 设备到设备接入
}

# 读取数据
def load_data():
    user_positions = pd.read_csv('/Users/a/Documents/Projects/web_question/data_4/用户位置4.csv')
    task_flow = pd.read_csv('/Users/a/Documents/Projects/web_question/data_4/用户任务流4.csv')
    mbs_large_scale = pd.read_csv('/Users/a/Documents/Projects/web_question/data_4/MBS_1大规模衰减.csv')
    mbs_small_scale = pd.read_csv('/Users/a/Documents/Projects/web_question/data_4/MBS_1小规模瑞丽衰减.csv')
    sbs_large_scale = [pd.read_csv(f'/Users/a/Documents/Projects/web_question/data_4/SBS_{i}大规模衰减.csv') for i in range(1, NUM_SBS+1)]
    sbs_small_scale = [pd.read_csv(f'/Users/a/Documents/Projects/web_question/data_4/SBS_{i}小规模瑞丽衰减.csv') for i in range(1, NUM_SBS+1)]
    return user_positions, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale

# 计算信道增益
def calculate_channel_gain(large_scale, small_scale, time_index, user_id, bs_id, power_dbm):
    # 获取用户列名
    user_column = USER_COLUMN_MAP[user_id]
    print(f'user_id: {user_id}, user_column: {user_column}, bs_id: {bs_id}')
    
    if bs_id == 0:  # MBS
        print(f'Accessing MBS data')
        large = large_scale.iloc[time_index][user_column]
        small = small_scale.iloc[time_index][user_column]
    else:  # SBS
        print(f'Accessing SBS data')
        large = large_scale.iloc[time_index][user_column]
        small = small_scale.iloc[time_index][user_column]
    
    # 根据附录公式计算接收功率 (mW)
    rx_power = 10 ** ((power_dbm - large) / 10) * (small ** 2)
    return rx_power

# 计算信干噪比和传输速率
def calculate_sinr_and_rate(access_decision, resource_allocation, power_control, time_index, 
                           mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale):
    b = 360000  # Hz, 每个资源块带宽
    NF = 7      # 噪声系数
    N0 = -174 + 10 * np.log10(b) + NF  # 白噪声功率谱密度 (dBm/Hz)
    
    sinr_values = np.zeros(NUM_USERS)
    rates = np.zeros(NUM_USERS)
    
    for u in range(NUM_USERS):
        bs = int(access_decision[u])
        if bs >= NUM_BS or bs < 0:
            bs = 0  # 如果索引超出范围或为负数，默认使用第一个基站
        allocated_blocks = int(resource_allocation[u, bs])
        if allocated_blocks == 0:
            continue
            
        # 计算有用信号功率
        power_dbm = power_control[bs]
        signal_power = calculate_channel_gain(
            mbs_large_scale if bs == 0 else sbs_large_scale[bs-1],
            mbs_small_scale if bs == 0 else sbs_small_scale[bs-1],
            time_index, u, bs, power_dbm
        )
        
        # 计算干扰功率
        interference_power = 0
        for other_bs in range(NUM_BS):
            if other_bs != bs and resource_allocation[u, other_bs] > 0:
                other_power_dbm = power_control[other_bs]
                other_signal = calculate_channel_gain(
                    mbs_large_scale if other_bs == 0 else sbs_large_scale[other_bs-1],
                    mbs_small_scale if other_bs == 0 else sbs_small_scale[other_bs-1],
                    time_index, u, other_bs, other_power_dbm
                )
                # 引入干扰协调机制：如果其他基站的信号强度过高，则降低其对当前用户的干扰影响
                interference_factor = 0.3 if other_signal > signal_power * 0.3 else 1.0  # 调整干扰协调参数
                interference_power += other_signal * interference_factor
        
        # 计算SINR
        noise_power = 10 ** (N0 / 10) * allocated_blocks * b
        sinr = signal_power / (interference_power + noise_power)
        sinr_values[u] = sinr
        
        # 计算传输速率
        rates[u] = allocated_blocks * b * np.log2(1 + sinr)
    
    return sinr_values, rates

# 计算用户服务质量
def calculate_qos(user_type, rate, latency, allocated_blocks):
    # SLA 参数 (根据附录表1)
    if user_type == 'URLLC':
        SLA_LATENCY = 1      # ms
        SLA_RATE = 10e6      # 10 Mbps
        PENALTY = -5
        ALPHA = 0.95
        
        if latency <= SLA_LATENCY:
            return ALPHA
        else:
            return PENALTY
            
    elif user_type == 'eMBB':
        SLA_LATENCY = 4      # ms
        SLA_RATE = 50e6      # 50 Mbps
        PENALTY = -3
        
        if rate >= SLA_RATE and latency <= SLA_LATENCY:
            return 1.0
        elif latency <= SLA_LATENCY:
            return rate / SLA_RATE
        else:
            return PENALTY
            
    else:  # mMTC
        SLA_LATENCY = 10     # ms
        PENALTY = -1
        
        if latency <= SLA_LATENCY:
            return allocated_blocks / 10  # 基于连接性比例
        else:
            return PENALTY

# 修改奖励函数为加权形式以平衡QoS和能耗
def calculate_reward(qos, energy, alpha=0.7, beta=0.3):
    # 奖励函数：R = α * QoS - β * Energy
    reward = alpha * qos - beta * energy
    return reward

# 扩展动作空间以适应新变量
def define_action_space():
    # 动作空间包括接入决策、资源分配、功率控制和接入模式
    action_space = spaces.Dict({
        'access_decision': spaces.MultiDiscrete([NUM_BS] * NUM_USERS),
        'resource_allocation': spaces.MultiDiscrete([MBS_RESOURCE_BLOCKS if i == 0 else SBS_RESOURCE_BLOCKS for i in range(NUM_BS)] * NUM_USERS),
        'power_control': spaces.Box(low=np.array([10, 10, 10, 10]), high=np.array([40, 30, 30, 30]), dtype=np.float32),
        'access_mode': spaces.MultiDiscrete([len(ACCESS_MODES)] * NUM_USERS)
    })
    return action_space

# 更新状态空间以包含宏基站和微基站的负载和信道信息
def define_state_space():
    # 状态空间包括用户位置、任务流、基站负载和信道信息
    state_space = spaces.Dict({
        'user_positions': spaces.Box(low=-1000, high=1000, shape=(NUM_USERS, 2), dtype=np.float32),
        'task_flow': spaces.Box(low=0, high=100, shape=(NUM_USERS,), dtype=np.float32),
        'bs_load': spaces.Box(low=0, high=100, shape=(NUM_BS,), dtype=np.float32),
        'channel_info': spaces.Box(low=0, high=1e6, shape=(NUM_BS, NUM_USERS), dtype=np.float32)
    })
    return state_space

# 在第四题的DRL模型上进行微调
def fine_tune_drl_model(model_path=None):
    """
    加载第四题的预训练模型并进行微调以适应能耗优化目标
    """
    class DRLAgent(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(DRLAgent, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    # 定义输入输出维度 (基于状态空间和动作空间)
    input_dim = NUM_USERS * 2 + NUM_USERS + NUM_BS + NUM_BS * NUM_USERS  # user_positions, task_flow, bs_load, channel_info
    output_dim = NUM_USERS * (NUM_BS + (MBS_RESOURCE_BLOCKS if NUM_BS == 1 else SBS_RESOURCE_BLOCKS) + len(ACCESS_MODES)) + NUM_BS  # access_decision, resource_allocation, access_mode, power_control
    
    # 初始化模型
    agent = DRLAgent(input_dim, output_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

    # 如果有预训练模型路径，加载预训练模型
    if model_path:
        try:
            agent.load_state_dict(torch.load(model_path))
            print(f"已加载预训练模型: {model_path}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("将从头开始训练模型")
    else:
        print("未提供预训练模型路径，将从头开始训练")

    return agent, optimizer

# 能耗计算函数
def calculate_energy(power_control):
    """
    计算基站的总能耗，基于功率控制值
    """
    # 假设能耗与功率成正比，转换为实际功率 (mW)
    total_energy = sum(10 ** (p / 10) for p in power_control)
    return total_energy

# 环境定义
class NetworkEnergyEnv(gym.Env):
    def __init__(self):
        super(NetworkEnergyEnv, self).__init__()
        self.action_space = define_action_space()
        self.observation_space = define_state_space()
        self.time_index = 0
        self.max_steps = 1000  # 最大仿真步数
        self.current_step = 0
        
        # 加载数据
        self.user_positions, self.task_flow, self.mbs_large_scale, self.mbs_small_scale, self.sbs_large_scale, self.sbs_small_scale = load_data()
        
    def reset(self):
        self.current_step = 0
        self.time_index = 0
        return self._get_state()
    
    def step(self, action):
        access_decision = action['access_decision']
        resource_allocation = action['resource_allocation']
        power_control = action['power_control']
        access_mode = action['access_mode']
        
        # 计算SINR和速率
        sinr_values, rates = calculate_sinr_and_rate(access_decision, resource_allocation, power_control, self.time_index, 
                                                    self.mbs_large_scale, self.mbs_small_scale, self.sbs_large_scale, self.sbs_small_scale)
        
        # 计算QoS
        total_qos = 0
        for u in range(NUM_USERS):
            bs = int(access_decision[u])
            if bs >= NUM_BS or bs < 0:
                bs = 0  # 如果索引超出范围或为负数，默认使用第一个基站
            allocated_blocks = int(resource_allocation[u][bs])
            if allocated_blocks == 0:
                continue
            user_type = USER_TYPES[u]
            try:
                time_row = self.task_flow.iloc[self.time_index]
                task_key = USER_COLUMN_MAP[u]
                queue_length = time_row.get(task_key, 1)
                bs_processing_factor = 1.0 if bs == 0 else 1.2
                user_type_factor = 0.8 if user_type == 'URLLC' else (0.9 if user_type == 'eMBB' else 1.0)
                latency = queue_length * 0.1 * bs_processing_factor * user_type_factor
            except:
                latency = 1.0
            qos = calculate_qos(user_type, rates[u], latency, allocated_blocks)
            weight = 1.5 if user_type == 'URLLC' else (1.2 if user_type == 'eMBB' else 1.0)
            total_qos += weight * qos
        
        # 计算能耗
        energy = calculate_energy(power_control)
        
        # 计算奖励
        reward = calculate_reward(total_qos, energy, alpha=0.7, beta=0.3)
        
        # 更新状态
        self.current_step += 1
        self.time_index = (self.time_index + 1) % len(self.task_flow)
        next_state = self._get_state()
        
        # 判断是否结束
        done = self.current_step >= self.max_steps
        
        # 返回额外信息用于记录
        info = {'qos': total_qos, 'energy': energy}
        
        return next_state, reward, done, info
    
    def _get_state(self):
        # 获取当前状态
        try:
            positions = self.user_positions.iloc[self.time_index].values.reshape(NUM_USERS, 2)
        except:
            positions = np.zeros((NUM_USERS, 2))
        
        try:
            task_flow_vals = self.task_flow.iloc[self.time_index].values
            task_vals = np.zeros(NUM_USERS)
            for u in range(NUM_USERS):
                task_key = USER_COLUMN_MAP[u]
                task_vals[u] = task_flow_vals[task_flow_vals == task_key].index[0] if task_key in task_flow_vals else 1
        except:
            task_vals = np.ones(NUM_USERS)
        
        bs_load = np.random.rand(NUM_BS) * 100  # 随机负载，实际应基于资源分配计算
        channel_info = np.random.rand(NUM_BS, NUM_USERS) * 1e5  # 随机信道信息，实际应基于信道增益计算
        
        return {
            'user_positions': positions.astype(np.float32),
            'task_flow': task_vals.astype(np.float32),
            'bs_load': bs_load.astype(np.float32),
            'channel_info': channel_info.astype(np.float32)
        }

# 训练和优化函数
def train_drl_model(agent, optimizer, env, num_episodes=100):
    """
    训练或微调DRL模型
    """
    # 用于记录训练过程中的奖励值、QoS 和能耗
    episode_rewards = []
    episode_qos = []
    episode_energy = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_qos = 0
        total_energy = 0
        done = False
        step = 0
        
        while not done:
            # 将状态转换为张量
            state_tensor = torch.tensor(np.concatenate([
                state['user_positions'].flatten(),
                state['task_flow'],
                state['bs_load'],
                state['channel_info'].flatten()
            ])).float()
            
            # 获取动作
            action_raw = agent(state_tensor)
            
            # 解析动作
            action = {}
            idx = 0
            action['access_decision'] = action_raw[idx:idx+NUM_USERS].argmax(dim=-1).detach().numpy() if action_raw[idx:idx+NUM_USERS].dim() > 1 else action_raw[idx:idx+NUM_USERS].detach().numpy()
            idx += NUM_USERS
            res_alloc_size = NUM_USERS * NUM_BS
            action['resource_allocation'] = action_raw[idx:idx+res_alloc_size].reshape(NUM_USERS, NUM_BS).argmax(dim=-1).detach().numpy() if action_raw[idx:idx+res_alloc_size].dim() > 1 else action_raw[idx:idx+res_alloc_size].reshape(NUM_USERS, NUM_BS).detach().numpy()
            idx += res_alloc_size
            action['power_control'] = action_raw[idx:idx+NUM_BS].clamp(10, 40).detach().numpy()
            idx += NUM_BS
            action['access_mode'] = action_raw[idx:idx+NUM_USERS].argmax(dim=-1).detach().numpy() if action_raw[idx:idx+NUM_USERS].dim() > 1 else action_raw[idx:idx+NUM_USERS].detach().numpy()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # 记录每一步的 QoS 和能耗 (假设环境返回这些信息)
            # 这里我们假设环境返回了这些值，实际中需要修改环境以返回这些信息
            total_qos += info.get('qos', 0)
            total_energy += info.get('energy', 0)
            
            # 简单的PPO近似更新 (实际应使用更完整的PPO或QMIX实现)
            if step % 10 == 0:
                optimizer.zero_grad()
                action_pred = agent(state_tensor)
                loss = -torch.tensor(reward, requires_grad=True).float()  # 简化损失函数，实际应使用PPO损失
                loss.backward()
                optimizer.step()
            
            state = next_state
            step += 1
        
        # 记录每个 episode 的数据
        episode_rewards.append(total_reward)
        episode_qos.append(total_qos)
        episode_energy.append(total_energy)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Total QoS: {total_qos}, Total Energy: {total_energy}")
    
    # 绘制并保存图表
    plot_training_metrics(episode_rewards, episode_qos, episode_energy)
    
    return agent

# 绘制训练过程中的奖励值、QoS 和能耗变化
def plot_training_metrics(rewards, qos_values, energy_values):
    episodes = range(len(rewards))
    
    plt.figure(figsize=(15, 5))
    
    # 绘制奖励值变化
    plt.subplot(1, 3, 1)
    plt.plot(episodes, rewards, 'b-', label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Over Episodes')
    plt.legend()
    plt.grid(True)
    
    # 绘制 QoS 变化
    plt.subplot(1, 3, 2)
    plt.plot(episodes, qos_values, 'g-', label='QoS')
    plt.xlabel('Episode')
    plt.ylabel('Total QoS')
    plt.title('Training QoS Over Episodes')
    plt.legend()
    plt.grid(True)
    
    # 绘制能耗变化
    plt.subplot(1, 3, 3)
    plt.plot(episodes, energy_values, 'r-', label='Energy')
    plt.xlabel('Episode')
    plt.ylabel('Total Energy')
    plt.title('Training Energy Over Episodes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/a/Documents/Projects/web_question/q5/training_metrics.png')
    plt.close()
    print("训练过程中的奖励值、QoS 和能耗变化图表已保存至 training_metrics.png")

if __name__ == '__main__':
    # 主程序逻辑
    action_space = define_action_space()
    state_space = define_state_space()
    
    # 初始化环境
    env = NetworkEnergyEnv()
    
    # 初始化和微调DRL模型
    agent, optimizer = fine_tune_drl_model()  # 可传入第四题模型路径，如 model_path='path_to_q4_model.pth'
    
    # 运行模拟和优化过程
    trained_agent = train_drl_model(agent, optimizer, env, num_episodes=100)
    
    print("训练完成！")
    
    # 保存模型
    torch.save(trained_agent.state_dict(), '/Users/a/Documents/Projects/web_question/q5/q5_trained_model.pth')
    print("模型已保存至 q5_trained_model.pth")