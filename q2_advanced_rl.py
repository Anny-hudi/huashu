import pandas as pd
import math
import numpy as np
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    """深度Q网络"""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class Task:
    """任务类"""
    def __init__(self, user_id, slice_type, task_size, arrival_time, x, y, large_scale, small_scale):
        self.user_id = user_id
        self.slice_type = slice_type
        self.task_size = task_size  # Mbit
        self.arrival_time = arrival_time
        self.x = x
        self.y = y
        self.large_scale = large_scale
        self.small_scale = small_scale
        self.remaining_size = task_size
        self.processing_time = 0
        self.completed = False

class AdvancedRLOptimizer:
    """高级强化学习优化器 - 修正版本"""
    
    def __init__(self):
        # 系统参数
        self.R_total = 50
        self.power = 30
        self.bandwidth_per_rb = 360e3
        self.thermal_noise = -174
        self.NF = 7
        
        # SLA参数 - 完全符合body_and_more.md表1
        self.URLLC_SLA_delay = 5
        self.eMBB_SLA_delay = 100
        self.mMTC_SLA_delay = 500
        self.URLLC_SLA_rate = 10
        self.eMBB_SLA_rate = 50
        self.mMTC_SLA_rate = 1
        
        # 惩罚系数 - 完全符合body_and_more.md表1
        self.M_URLLC = 5
        self.M_eMBB = 3
        self.M_mMTC = 1
        self.alpha = 0.95
        
        # 用户数量
        self.URLLC_users = 2
        self.eMBB_users = 4
        self.mMTC_users = 10
        
        # 资源块占用量约束 - 根据body_and_more.md表1
        self.URLLC_rb_per_user = 10  # 每个URLLC用户需要10个资源块
        self.eMBB_rb_per_user = 5    # 每个eMBB用户需要5个资源块
        self.mMTC_rb_per_user = 2    # 每个mMTC用户需要2个资源块
        
        # 满足倍数约束的资源分配方案
        self.urllc_possible = [0, 10, 20, 30, 40, 50]  # 10的倍数
        self.embb_possible = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # 5的倍数
        self.mmtc_possible = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]  # 2的倍数
        
        # 强化学习参数
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # 网络参数
        self.state_size = 8  # 增加状态维度
        self.action_size = len(self.urllc_possible) * len(self.embb_possible) * len(self.mmtc_possible)
        
        # DQN网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.dqn_optimizer = optim.Adam(self.dqn_network.parameters(), lr=self.learning_rate)
        
        # 策略网络
        self.policy_network = PolicyNetwork(self.state_size, self.action_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        # 用户映射
        self.user_mapping = {
            'URLLC': ['U1', 'U2'],
            'eMBB': ['e1', 'e2', 'e3', 'e4'],
            'mMTC': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
        }
        
        # 任务队列 - 按切片类型分别管理
        self.task_queues = {
            'URLLC': deque(),
            'eMBB': deque(),
            'mMTC': deque()
        }
        
        # 任务到达分布参数
        self.urllc_lambda = 0.3  # 泊松分布参数
        self.embb_uniform_prob = 0.4  # 均匀分布概率
        self.mmtc_uniform_prob = 0.6  # 均匀分布概率
        
        # 更新目标网络
        self.update_target_network()
        
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.dqn_network.state_dict())
    
    def generate_task_arrival(self, time_point, user_data):
        """实现任务到达分布 - 符合body_and_more.md表1要求"""
        new_tasks = []
        
        # URLLC: 泊松分布
        for user in self.user_mapping['URLLC']:
            if user in user_data and user_data[user]['task_size'] > 0:
                # 泊松分布：P(X=k) = (λ^k * e^(-λ)) / k!
                arrival_prob = np.random.poisson(self.urllc_lambda) / 10  # 归一化概率
                if random.random() < arrival_prob:
                    task = Task(
                        user_id=user,
                        slice_type='URLLC',
                        task_size=user_data[user]['task_size'],
                        arrival_time=time_point,
                        x=user_data[user]['x'],
                        y=user_data[user]['y'],
                        large_scale=user_data[user].get('large_scale', 0),
                        small_scale=user_data[user].get('small_scale', 1)
                    )
                    new_tasks.append(task)
        
        # eMBB: 均匀分布
        for user in self.user_mapping['eMBB']:
            if user in user_data and user_data[user]['task_size'] > 0:
                if random.random() < self.embb_uniform_prob:
                    task = Task(
                        user_id=user,
                        slice_type='eMBB',
                        task_size=user_data[user]['task_size'],
                        arrival_time=time_point,
                        x=user_data[user]['x'],
                        y=user_data[user]['y'],
                        large_scale=user_data[user].get('large_scale', 0),
                        small_scale=user_data[user].get('small_scale', 1)
                    )
                    new_tasks.append(task)
        
        # mMTC: 均匀分布
        for user in self.user_mapping['mMTC']:
            if user in user_data and user_data[user]['task_size'] > 0:
                if random.random() < self.mmtc_uniform_prob:
                    task = Task(
                        user_id=user,
                        slice_type='mMTC',
                        task_size=user_data[user]['task_size'],
                        arrival_time=time_point,
                        x=user_data[user]['x'],
                        y=user_data[user]['y'],
                        large_scale=user_data[user].get('large_scale', 0),
                        small_scale=user_data[user].get('small_scale', 1)
                    )
                    new_tasks.append(task)
        
        return new_tasks
    
    def update_user_mobility(self, user_data, time_point):
        """更新用户移动性 - 考虑信道变化"""
        # 模拟用户移动对信道的影响
        mobility_factor = 1 + 0.1 * math.sin(time_point / 1000 * 2 * math.pi)  # 周期性变化
        
        for user, info in user_data.items():
            if 'large_scale' in info:
                # 大规模衰减受移动性影响
                info['large_scale'] *= mobility_factor
            if 'small_scale' in info:
                # 小规模瑞丽衰减受移动性影响
                info['small_scale'] *= (0.8 + 0.4 * random.random())  # 随机变化
        
        return user_data
    
    def calculate_sinr(self, power_dbm, large_scale_db, small_scale, user_x, user_y, num_rbs):
        """计算信干噪比 - 考虑用户移动性"""
        power_mw = 10**((power_dbm - 30) / 10)
        distance_m = math.sqrt(user_x**2 + user_y**2)
        distance_km = distance_m / 1000
        frequency_ghz = 2.6
        distance_path_loss_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 147.55
        small_scale_positive = max(small_scale, 0.001)
        total_channel_gain_db = large_scale_db + 10 * math.log10(small_scale_positive) - distance_path_loss_db
        channel_gain_linear = 10**(total_channel_gain_db / 10)
        received_power = power_mw * channel_gain_linear
        noise_power = 10**((self.thermal_noise + 10*math.log10(num_rbs * self.bandwidth_per_rb) + self.NF) / 10)
        sinr = received_power / noise_power
        return sinr
    
    def calculate_rate(self, sinr, num_rbs):
        """计算传输速率 (Mbps)"""
        rate = num_rbs * self.bandwidth_per_rb * math.log2(1 + sinr)
        return rate / 1e6
    
    def calculate_urllc_qos(self, rate, delay):
        """计算URLLC服务质量 - 完全符合公式"""
        if delay <= self.URLLC_SLA_delay:
            return self.alpha ** delay  # α^L
        else:
            return -self.M_URLLC  # -5
    
    def calculate_embb_qos(self, rate, delay):
        """计算eMBB服务质量 - 完全符合公式"""
        if delay <= self.eMBB_SLA_delay:
            if rate >= self.eMBB_SLA_rate:
                return 1.0
            else:
                return rate / self.eMBB_SLA_rate
        else:
            return -self.M_eMBB  # -3
    
    def calculate_mmtc_qos(self, connection_ratio, delay):
        """计算mMTC服务质量 - 完全符合公式"""
        if delay <= self.mMTC_SLA_delay:
            return connection_ratio  # Σc_i' / Σc_i
        else:
            return -self.M_mMTC  # -1
    
    def get_state_representation(self, user_data, task_queues):
        """获取状态表示 - 增强版本"""
        state = []
        
        # 当前活跃任务数
        active_tasks = sum(1 for user, info in user_data.items() if info['task_size'] > 0)
        state.append(active_tasks / 16)
        
        # 各队列长度
        urllc_queue_len = len(task_queues['URLLC'])
        embb_queue_len = len(task_queues['eMBB'])
        mmtc_queue_len = len(task_queues['mMTC'])
        state.append(min(urllc_queue_len / 50, 1.0))
        state.append(min(embb_queue_len / 50, 1.0))
        state.append(min(mmtc_queue_len / 50, 1.0))
        
        # 各切片任务比例
        urllc_tasks = sum(1 for user in self.user_mapping['URLLC'] 
                         if user in user_data and user_data[user]['task_size'] > 0)
        embb_tasks = sum(1 for user in self.user_mapping['eMBB'] 
                        if user in user_data and user_data[user]['task_size'] > 0)
        mmtc_tasks = sum(1 for user in self.user_mapping['mMTC'] 
                        if user in user_data and user_data[user]['task_size'] > 0)
        
        state.append(urllc_tasks / 2)
        state.append(embb_tasks / 4)
        state.append(mmtc_tasks / 10)
        
        # 平均信道质量
        avg_channel_quality = 0
        count = 0
        for user, info in user_data.items():
            if info['task_size'] > 0:
                large_scale = info.get('large_scale', 0)
                small_scale = info.get('small_scale', 1)
                avg_channel_quality += (large_scale + 10 * math.log10(max(small_scale, 0.001)))
                count += 1
        
        if count > 0:
            avg_channel_quality /= count
            state.append((avg_channel_quality + 100) / 200)
        else:
            state.append(0.5)
        
        return torch.FloatTensor(state).to(self.device)
    
    def action_to_allocation(self, action):
        """将动作转换为满足倍数约束的资源分配"""
        # 计算所有可能的分配组合
        valid_allocations = []
        for urllc_rbs in self.urllc_possible:
            for embb_rbs in self.embb_possible:
                for mmtc_rbs in self.mmtc_possible:
                    if urllc_rbs + embb_rbs + mmtc_rbs <= self.R_total:
                        valid_allocations.append((urllc_rbs, embb_rbs, mmtc_rbs))
        
        if action >= len(valid_allocations):
            action = action % len(valid_allocations)
        
        return valid_allocations[action]
    
    def process_tasks(self, resource_allocation, time_slice):
        """完整的任务处理流程 - 符合user.md要求"""
        urllc_rbs, embb_rbs, mmtc_rbs = resource_allocation
        total_qos = 0
        
        # 处理URLLC任务
        if urllc_rbs > 0 and self.task_queues['URLLC']:
            urllc_rb_per_user = urllc_rbs / self.URLLC_users
            for _ in range(min(len(self.task_queues['URLLC']), self.URLLC_users)):
                if self.task_queues['URLLC']:
                    task = self.task_queues['URLLC'][0]
                    
                    # 计算传输速率
                    sinr = self.calculate_sinr(self.power, task.large_scale, task.small_scale, 
                                             task.x, task.y, urllc_rb_per_user)
                    rate = self.calculate_rate(sinr, urllc_rb_per_user)
                    
                    # 计算传输量
                    transmission_size = rate * time_slice / 1000  # Mbit
                    task.remaining_size -= transmission_size
                    task.processing_time += time_slice
                    
                    # 检查任务是否完成
                    if task.remaining_size <= 0:
                        task.completed = True
                        delay = task.processing_time
                        qos = self.calculate_urllc_qos(rate, delay)
                        total_qos += qos
                        self.task_queues['URLLC'].popleft()
                    else:
                        # 任务未完成，继续排队
                        break
        
        # 处理eMBB任务
        if embb_rbs > 0 and self.task_queues['eMBB']:
            embb_rb_per_user = embb_rbs / self.eMBB_users
            for _ in range(min(len(self.task_queues['eMBB']), self.eMBB_users)):
                if self.task_queues['eMBB']:
                    task = self.task_queues['eMBB'][0]
                    
                    # 计算传输速率
                    sinr = self.calculate_sinr(self.power, task.large_scale, task.small_scale, 
                                             task.x, task.y, embb_rb_per_user)
                    rate = self.calculate_rate(sinr, embb_rb_per_user)
                    
                    # 计算传输量
                    transmission_size = rate * time_slice / 1000  # Mbit
                    task.remaining_size -= transmission_size
                    task.processing_time += time_slice
                    
                    # 检查任务是否完成
                    if task.remaining_size <= 0:
                        task.completed = True
                        delay = task.processing_time
                        qos = self.calculate_embb_qos(rate, delay)
                        total_qos += qos
                        self.task_queues['eMBB'].popleft()
                    else:
                        # 任务未完成，继续排队
                        break
        
        # 处理mMTC任务
        if mmtc_rbs > 0 and self.task_queues['mMTC']:
            mmtc_rb_per_user = mmtc_rbs / self.mMTC_users
            connected_users = 0
            total_users = len(self.task_queues['mMTC'])
            
            for _ in range(min(len(self.task_queues['mMTC']), self.mMTC_users)):
                if self.task_queues['mMTC']:
                    task = self.task_queues['mMTC'][0]
                    
                    # 计算传输速率
                    sinr = self.calculate_sinr(self.power, task.large_scale, task.small_scale, 
                                             task.x, task.y, mmtc_rb_per_user)
                    rate = self.calculate_rate(sinr, mmtc_rb_per_user)
                    
                    # 检查连接成功
                    if rate >= self.mMTC_SLA_rate:
                        connected_users += 1
                    
                    # 计算传输量
                    transmission_size = rate * time_slice / 1000  # Mbit
                    task.remaining_size -= transmission_size
                    task.processing_time += time_slice
                    
                    # 检查任务是否完成
                    if task.remaining_size <= 0:
                        task.completed = True
                        self.task_queues['mMTC'].popleft()
                    else:
                        # 任务未完成，继续排队
                        break
            
            # 计算mMTC QoS
            connection_ratio = connected_users / total_users if total_users > 0 else 0
            delay = 100  # 假设延迟
            qos = self.calculate_mmtc_qos(connection_ratio, delay)
            total_qos += qos
        
        return total_qos
    
    def evaluate_allocation(self, urllc_rbs, embb_rbs, mmtc_rbs, user_data, task_queues):
        """评估资源分配方案的QoS - 使用完整任务处理流程"""
        # 将新任务加入队列
        new_tasks = self.generate_task_arrival(0, user_data)  # 时间点设为0
        for task in new_tasks:
            self.task_queues[task.slice_type].append(task)
        
        # 使用完整任务处理流程
        total_qos = self.process_tasks((urllc_rbs, embb_rbs, mmtc_rbs), 100)  # 100ms时间片
        
        return total_qos
    
    def dqn_optimization(self, user_data, task_queues):
        """DQN优化"""
        state = self.get_state_representation(user_data, task_queues)
        
        # ε-贪婪策略
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn_network(state.unsqueeze(0))
                action = q_values.argmax().item()
        
        # 执行动作
        allocation = self.action_to_allocation(action)
        qos = self.evaluate_allocation(*allocation, user_data, task_queues)
        
        # 存储经验
        next_state = self.get_state_representation(user_data, task_queues)
        self.memory.append((state, action, qos, next_state))
        
        # 训练网络
        if len(self.memory) >= self.batch_size:
            self.train_dqn()
        
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return allocation, qos
    
    def train_dqn(self):
        """训练DQN网络"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        
        current_q_values = self.dqn_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.discount_factor * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
    
    def policy_gradient_optimization(self, user_data, task_queues):
        """策略梯度优化"""
        state = self.get_state_representation(user_data, task_queues)
        
        # 获取动作概率
        action_probs = self.policy_network(state.unsqueeze(0))
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # 执行动作
        allocation = self.action_to_allocation(action.item())
        qos = self.evaluate_allocation(*allocation, user_data, task_queues)
        
        # 计算损失
        log_prob = action_dist.log_prob(action)
        loss = -log_prob * qos  # 策略梯度损失
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        return allocation, qos
    
    def hybrid_rl_optimization(self, user_data, task_queues):
        """混合强化学习优化"""
        # DQN优化
        dqn_allocation, dqn_qos = self.dqn_optimization(user_data, task_queues)
        
        # 策略梯度优化
        pg_allocation, pg_qos = self.policy_gradient_optimization(user_data, task_queues)
        
        # 选择最佳结果
        if dqn_qos > pg_qos:
            return dqn_allocation, dqn_qos, "DQN"
        else:
            return pg_allocation, pg_qos, "Policy Gradient"

def solve_problem_2_advanced_rl():
    """使用高级强化学习解决第二问 - 修正版本"""
    
    # 初始化优化器
    optimizer = AdvancedRLOptimizer()
    
    # 时间参数 - 符合问题2要求
    total_time = 1000  # 1000ms
    decision_interval = 100  # 每100ms决策一次
    num_decisions = total_time // decision_interval  # 10次决策
    
    print(f"=== 第二问：高级强化学习解决方案（修正版本）===")
    print(f"总时间: {total_time}ms, 决策间隔: {decision_interval}ms, 决策次数: {num_decisions}")
    print(f"资源块倍数约束: URLLC(10), eMBB(5), mMTC(2)")
    print(f"任务到达分布: URLLC(泊松), eMBB(均匀), mMTC(均匀)")
    print(f"用户移动性: 已考虑信道变化")
    
    # 加载数据
    print("\n=== 加载数据 ===")
    task_flow_data = pd.read_csv('data_2/用户任务流2.csv')
    user_position_data = pd.read_csv('data_2/用户位置2.csv')
    large_scale_data = pd.read_csv('data_2/大规模衰减2.csv')
    small_scale_data = pd.read_csv('data_2/小规模瑞丽衰减2.csv')
    
    print(f"任务流数据形状: {task_flow_data.shape}")
    print(f"用户位置数据形状: {user_position_data.shape}")
    print(f"大规模衰减数据形状: {large_scale_data.shape}")
    print(f"小规模瑞丽衰减数据形状: {small_scale_data.shape}")
    
    def get_user_data_at_time(time_idx):
        """获取指定时间点的用户数据 - 考虑移动性"""
        time_point = time_idx * decision_interval / 1000
        
        time_diff = abs(task_flow_data['Time'] - time_point)
        closest_idx = time_diff.idxmin()
        
        user_data = {}
        
        for slice_type, users in optimizer.user_mapping.items():
            for user in users:
                task_size = task_flow_data.loc[closest_idx, user]
                user_data[user] = {
                    'slice_type': slice_type,
                    'task_size': task_size,
                    'time': time_point
                }
        
        for slice_type, users in optimizer.user_mapping.items():
            for user in users:
                x_col = f"{user}_X"
                y_col = f"{user}_Y"
                if x_col in user_position_data.columns and y_col in user_position_data.columns:
                    user_data[user]['x'] = user_position_data.loc[closest_idx, x_col]
                    user_data[user]['y'] = user_position_data.loc[closest_idx, y_col]
        
        for slice_type, users in optimizer.user_mapping.items():
            for user in users:
                if user in large_scale_data.columns:
                    user_data[user]['large_scale'] = large_scale_data.loc[closest_idx, user]
                if user in small_scale_data.columns:
                    user_data[user]['small_scale'] = small_scale_data.loc[closest_idx, user]
        
        # 更新用户移动性
        user_data = optimizer.update_user_mobility(user_data, time_point)
        
        return user_data
    
    # 主优化循环
    print(f"\n=== 开始高级强化学习优化 ===")
    
    all_allocations = []
    
    for decision_idx in range(num_decisions):
        print(f"\n--- 第 {decision_idx + 1} 次决策 (时间: {decision_idx * decision_interval}ms) ---")
        
        user_data = get_user_data_at_time(decision_idx)
        
        active_tasks = 0
        for user, info in user_data.items():
            if info['task_size'] > 0:
                active_tasks += 1
        
        print(f"当前时间点活跃任务数: {active_tasks}")
        print(f"URLLC队列长度: {len(optimizer.task_queues['URLLC'])}")
        print(f"eMBB队列长度: {len(optimizer.task_queues['eMBB'])}")
        print(f"mMTC队列长度: {len(optimizer.task_queues['mMTC'])}")
        print(f"探索率: {optimizer.epsilon:.4f}")
        
        # 混合强化学习优化
        best_allocation, best_qos, method = optimizer.hybrid_rl_optimization(user_data, optimizer.task_queues)
        
        if best_allocation:
            urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
            print(f"最优分配 ({method}): URLLC={urllc_rbs}, eMBB={embb_rbs}, mMTC={mmtc_rbs}")
            print(f"总QoS: {best_qos:.4f}")
            
            # 验证倍数约束
            urllc_valid = urllc_rbs % 10 == 0
            embb_valid = embb_rbs % 5 == 0
            mmtc_valid = mmtc_rbs % 2 == 0
            total_valid = urllc_rbs + embb_rbs + mmtc_rbs <= 50
            
            print(f"约束检查: URLLC倍数({urllc_valid}), eMBB倍数({embb_valid}), mMTC倍数({mmtc_valid}), 总量({total_valid})")
            
            all_allocations.append({
                'decision_idx': decision_idx,
                'time': decision_idx * decision_interval,
                'urllc_rbs': urllc_rbs,
                'embb_rbs': embb_rbs,
                'mmtc_rbs': mmtc_rbs,
                'total_qos': best_qos,
                'active_tasks': active_tasks,
                'method': method,
                'constraints_satisfied': urllc_valid and embb_valid and mmtc_valid and total_valid
            })
        else:
            print("未找到可行解")
            all_allocations.append({
                'decision_idx': decision_idx,
                'time': decision_idx * decision_interval,
                'urllc_rbs': 0,
                'embb_rbs': 0,
                'mmtc_rbs': 0,
                'total_qos': 0,
                'active_tasks': active_tasks,
                'method': 'None',
                'constraints_satisfied': False
            })
        
        # 定期更新目标网络
        if decision_idx % 5 == 0:
            optimizer.update_target_network()
    
    # 输出结果
    print(f"\n=== 最终结果 ===")
    print("决策时间序列资源分配方案:")
    for allocation in all_allocations:
        print(f"时间 {allocation['time']}ms: URLLC={allocation['urllc_rbs']}, "
              f"eMBB={allocation['embb_rbs']}, mMTC={allocation['mmtc_rbs']}, "
              f"QoS={allocation['total_qos']:.4f}, 活跃任务={allocation['active_tasks']}, "
              f"方法={allocation['method']}, 约束满足={allocation['constraints_satisfied']}")
    
    total_qos = sum(allocation['total_qos'] for allocation in all_allocations)
    avg_qos = total_qos / len(all_allocations)
    constraints_satisfied_count = sum(1 for allocation in all_allocations if allocation['constraints_satisfied'])
    
    print(f"\n总体性能:")
    print(f"总QoS: {total_qos:.4f}")
    print(f"平均QoS: {avg_qos:.4f}")
    print(f"约束满足次数: {constraints_satisfied_count}/{len(all_allocations)}")
    
    method_counts = {}
    for allocation in all_allocations:
        method = allocation['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\n算法使用统计:")
    for method, count in method_counts.items():
        print(f"{method}: {count}次")
    
    return all_allocations

if __name__ == "__main__":
    solve_problem_2_advanced_rl() 