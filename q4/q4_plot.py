# 第4题解决方案：异构网络中的接入决策、资源分配和功率控制

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt

# 数据目录与结果保存路径（Windows）
DATA_DIR = r"C:\Users\Anny\PycharmProjects\huashu\data_4"
RESULTS_PATH = r"C:\Users\Anny\PycharmProjects\huashu\q4\results.csv"

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

# 用户类型到索引映射（用于统计）
TYPE_TO_INDEX = {'URLLC': 0, 'eMBB': 1, 'mMTC': 2}
INDEX_TO_TYPE = {0: 'URLLC', 1: 'eMBB', 2: 'mMTC'}

# 用户索引到列名的映射
USER_COLUMN_MAP = {}
for i in range(10):  # U1-U10: URLLC
    USER_COLUMN_MAP[i] = f'U{i+1}'
for i in range(10, 30):  # e1-e20: eMBB
    USER_COLUMN_MAP[i] = f'e{i-9}'
for i in range(30, 70):  # m1-m40: mMTC
    USER_COLUMN_MAP[i] = f'm{i-29}'

# 读取数据
def load_data():
    user_positions = pd.read_csv(os.path.join(DATA_DIR, '用户位置4.csv'))
    task_flow = pd.read_csv(os.path.join(DATA_DIR, '用户任务流4.csv'))
    mbs_large_scale = pd.read_csv(os.path.join(DATA_DIR, 'MBS_1大规模衰减.csv'))
    mbs_small_scale = pd.read_csv(os.path.join(DATA_DIR, 'MBS_1小规模瑞丽衰减.csv'))
    sbs_large_scale = [pd.read_csv(os.path.join(DATA_DIR, f'SBS_{i}大规模衰减.csv')) for i in range(1, NUM_SBS+1)]
    sbs_small_scale = [pd.read_csv(os.path.join(DATA_DIR, f'SBS_{i}小规模瑞丽衰减.csv')) for i in range(1, NUM_SBS+1)]
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
        bs = access_decision[u]
        allocated_blocks = resource_allocation[bs, u]
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
            if other_bs != bs and resource_allocation[other_bs, u] > 0:
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

# 计算适应度函数 (用户服务质量)
def fitness(individual, time_index, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale):
    access_decision, resource_allocation = individual
    
    # 初始化功率控制 (dBm)
    power_control = np.array([30, 20, 20, 20])  # MBS: 30dBm, SBS: 20dBm
    
    # 计算SINR和传输速率
    sinr_values, rates = calculate_sinr_and_rate(
        access_decision, resource_allocation, power_control, time_index,
        mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale
    )
    
    total_qos = 0
    for u in range(NUM_USERS):
        bs = access_decision[u]
        allocated_blocks = resource_allocation[bs, u]
        
        if allocated_blocks == 0:
            continue
            
        # 获取用户类型
        user_type = USER_TYPES[u]
        
        # 简化时延计算 (基于任务队列长度)
        try:
            time_row = task_flow.iloc[time_index]
            task_key = USER_COLUMN_MAP[u]
            queue_length = time_row.get(task_key, 1)
            # 考虑基站处理能力和用户任务类型
            bs_processing_factor = 1.0 if bs == 0 else 1.2  # MBS处理能力较强，SBS稍弱
            user_type_factor = 0.8 if user_type == 'URLLC' else (0.9 if user_type == 'eMBB' else 1.0)
            latency = queue_length * 0.1 * bs_processing_factor * user_type_factor  # 假设每个任务0.1ms，调整后的时延
        except:
            latency = 1.0  # 默认时延
        
        # 计算QoS
        qos = calculate_qos(user_type, rates[u], latency, allocated_blocks)
        # 根据用户类型调整QoS权重
        weight = 1.0
        if user_type == 'URLLC':
            weight = 1.5  # URLLC用户更高的权重
        elif user_type == 'eMBB':
            weight = 1.2  # eMBB用户中等权重
        total_qos += weight * qos
    
    return total_qos

# 遗传算法优化接入决策和资源分配
def genetic_algorithm(time_index, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale, convergence_recorder=None):
    # 初始化种群
    population_size = 150  # 增加种群大小以提高优化效果
    population = []
    
    for _ in range(population_size):
        # 随机接入决策
        access_decision = np.random.randint(0, NUM_BS, size=NUM_USERS)
        
        # 资源分配矩阵 (基站 x 用户)
        resource_allocation = np.zeros((NUM_BS, NUM_USERS), dtype=int)
        
        # 资源块限制
        resource_limits = [100, 50, 50, 50]  # MBS: 100, SBS1-3: 50 each
        
        # 获取任务队列长度以确定优先级
        time_row = task_flow.iloc[time_index]
        user_priorities = []
        for u in range(NUM_USERS):
            task_key = USER_COLUMN_MAP[u]
            queue_length = time_row.get(task_key, 1)
            user_type = USER_TYPES[u]
            # 计算信道增益以进一步调整优先级
            bs = access_decision[u]
            power_dbm = 30 if bs == 0 else 20  # 假设初始功率
            channel_gain = calculate_channel_gain(
                mbs_large_scale if bs == 0 else sbs_large_scale[bs-1],
                mbs_small_scale if bs == 0 else sbs_small_scale[bs-1],
                time_index, u, bs, power_dbm
            )
            # 综合考虑任务队列长度、用户类型和信道条件
            priority_score = queue_length * (1.5 if user_type == 'URLLC' else (1.2 if user_type == 'eMBB' else 1.0)) / (channel_gain + 1e-10)  # 避免除以零
            user_priorities.append((u, priority_score, user_type))
        
        # 根据综合优先级分数排序用户，分数高的优先
        user_priorities.sort(key=lambda x: -x[1])
        
        # 动态分配资源
        for u, _, user_type in user_priorities:
            bs = access_decision[u]
            if resource_limits[bs] > 0:
                if user_type in ['URLLC', 'eMBB']:
                    max_blocks = min(resource_limits[bs], 15)  # 增加URLLC和eMBB的最大资源块
                    blocks = np.random.randint(5, max_blocks + 1) if max_blocks >= 5 else max_blocks
                else:  # mMTC
                    max_blocks = min(resource_limits[bs], 5)
                    blocks = np.random.randint(0, max_blocks + 1)
                if blocks <= resource_limits[bs]:
                    resource_allocation[bs, u] = blocks
                    resource_limits[bs] -= blocks
                else:
                    blocks = resource_limits[bs]  # 确保不超过剩余资源
                    resource_allocation[bs, u] = blocks
                    resource_limits[bs] -= blocks
        
        individual = (access_decision, resource_allocation)
        population.append(individual)
    
    # 遗传算法迭代
    generations = 300  # 增加迭代次数以提高优化效果
    for gen in range(generations):
        # 评估适应度
        fitness_values = [fitness(ind, time_index, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale) for ind in population]
        # 记录收敛曲线（每代的最优适应度）
        if convergence_recorder is not None:
            try:
                convergence_recorder.append(max(fitness_values))
            except Exception:
                pass
        
        # 选择
        sorted_indices = np.argsort(fitness_values)[::-1]
        elite_size = 60  # 进一步增加精英个体数量以保留更多优秀解
        elite = [population[i] for i in sorted_indices[:elite_size]]
        
        # 交叉和变异
        new_population = elite.copy()
        while len(new_population) < population_size:
            # 选择父母
            parent1, parent2 = random.choices(elite, k=2)
            
            # 交叉
            child_access = np.where(np.random.rand(NUM_USERS) < 0.5, parent1[0], parent2[0])
            child_resource = np.where(np.random.rand(NUM_BS, NUM_USERS) < 0.5, parent1[1], parent2[1])
            
            # 变异
            if np.random.rand() < 0.3:  # 增加变异概率以探索更多解
                mutate_idx = np.random.randint(0, NUM_USERS)
                child_access[mutate_idx] = np.random.randint(0, NUM_BS)
            
            if np.random.rand() < 0.3:  # 增加变异概率以探索更多解
                # 重新计算资源限制以确保不超过限制
                resource_limits = [100, 50, 50, 50]
                for u in range(NUM_USERS):
                    bs = child_access[u]
                    blocks = child_resource[bs, u]
                    resource_limits[bs] -= blocks
                
                mutate_bs = np.random.randint(0, NUM_BS)
                mutate_user = np.random.randint(0, NUM_USERS)
                max_blocks = min(resource_limits[mutate_bs], 10 if USER_TYPES[mutate_user] in ['URLLC', 'eMBB'] else 5)
                if max_blocks > 0:
                    if USER_TYPES[mutate_user] in ['URLLC', 'eMBB']:
                        child_resource[mutate_bs, mutate_user] = np.random.randint(5, max_blocks + 1) if max_blocks >= 5 else max_blocks
                    else:
                        child_resource[mutate_bs, mutate_user] = np.random.randint(0, max_blocks + 1)
                else:
                    child_resource[mutate_bs, mutate_user] = 0
                
                # 调整其他用户的资源分配以确保总资源不超过限制
                for bs in range(NUM_BS):
                    if resource_limits[bs] < 0:
                        excess = -resource_limits[bs]
                        users_on_bs = [u for u in range(NUM_USERS) if child_access[u] == bs]
                        np.random.shuffle(users_on_bs)
                        for u in users_on_bs:
                            if excess <= 0:
                                break
                            if child_resource[bs, u] > 0:
                                reduction = min(excess, child_resource[bs, u])
                                child_resource[bs, u] -= reduction
                                excess -= reduction
                                resource_limits[bs] += reduction
            
            # 再次检查并调整资源分配以确保不超过限制
            resource_limits = [100, 50, 50, 50]
            for u in range(NUM_USERS):
                bs = child_access[u]
                blocks = child_resource[bs, u]
                resource_limits[bs] -= blocks
            for bs in range(NUM_BS):
                if resource_limits[bs] < 0:
                    excess = -resource_limits[bs]
                    users_on_bs = [u for u in range(NUM_USERS) if child_access[u] == bs]
                    np.random.shuffle(users_on_bs)
                    for u in users_on_bs:
                        if excess <= 0:
                            break
                        if child_resource[bs, u] > 0:
                            reduction = min(excess, child_resource[bs, u])
                            child_resource[bs, u] -= reduction
                            excess -= reduction
                            resource_limits[bs] += reduction
            
            new_population.append((child_access, child_resource))
        
        population = new_population[:population_size]
    
    # 返回最优个体
    fitness_values = [fitness(ind, time_index, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale) for ind in population]
    best_individual = population[np.argmax(fitness_values)]
    return best_individual

# 凸优化功率控制
def convex_optimization_power_control(access_decision, resource_allocation, time_index, 
                                    mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale, task_flow):
    # 功率范围约束 (dBm)
    power_min = np.array([10, 10, 10, 10])  # MBS和SBS最小功率
    power_max = np.array([40, 30, 30, 30])  # MBS最大40dBm, SBS最大30dBm
    
    # 目标函数：最大化总用户服务质量
    def objective(power_dbm):
        total_qos = 0
        sinr_values, rates = calculate_sinr_and_rate(
            access_decision, resource_allocation, power_dbm, time_index,
            mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale
        )
        
        # 计算每个基站的平均信道增益
        bs_channel_gains = np.zeros(NUM_BS)
        bs_user_counts = np.zeros(NUM_BS)
        for u in range(NUM_USERS):
            bs = access_decision[u]
            allocated_blocks = resource_allocation[bs, u]
            if allocated_blocks > 0:
                channel_gain = calculate_channel_gain(
                    mbs_large_scale if bs == 0 else sbs_large_scale[bs-1],
                    mbs_small_scale if bs == 0 else sbs_small_scale[bs-1],
                    time_index, u, bs, power_dbm[bs]
                )
                bs_channel_gains[bs] += channel_gain
                bs_user_counts[bs] += 1
        
        # 调整功率以补偿信道增益低的基站
        for bs in range(NUM_BS):
            if bs_user_counts[bs] > 0:
                avg_gain = bs_channel_gains[bs] / bs_user_counts[bs]
                # 如果平均信道增益低，则增加功率
                if avg_gain < 1e-5:  # 阈值可以调整
                    power_dbm[bs] = min(power_dbm[bs] + 2, power_max[bs])  # 增加2dBm，但不超过最大值
        
        for u in range(NUM_USERS):
            bs = access_decision[u]
            allocated_blocks = resource_allocation[bs, u]
            
            if allocated_blocks == 0:
                continue
                
            user_type = USER_TYPES[u]
            try:
                time_row = task_flow.iloc[time_index]
                task_key = USER_COLUMN_MAP[u]
                queue_length = time_row.get(task_key, 1)
                bs_processing_factor = 1.0 if bs == 0 else 1.2
                user_type_factor = 0.8 if user_type == 'URLLC' else (0.9 if user_type == 'eMBB' else 1.0)
                latency = queue_length * 0.1 * bs_processing_factor * user_type_factor
            except:
                latency = 1.0  # 默认时延
            
            # 根据用户类型和信道条件调整QoS权重
            weight = 1.0
            if user_type == 'URLLC':
                weight = 1.5  # URLLC用户更高的权重
            elif user_type == 'eMBB':
                weight = 1.2  # eMBB用户中等权重
            qos = calculate_qos(user_type, rates[u], latency, allocated_blocks)
            total_qos += weight * qos
        
        return -total_qos  # 最小化负值 = 最大化正值
    
    # 约束条件
    constraints = []
    
    # 功率范围约束
    for i in range(NUM_BS):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - power_min[i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: power_max[i] - x[i]})
    
    # 初始功率
    initial_power = np.array([30, 20, 20, 20])
    
    # 优化
    try:
        result = minimize(objective, initial_power, constraints=constraints, method='SLSQP')
        if result.success:
            return result.x
        else:
            return initial_power
    except:
        return initial_power

# 主函数
def main():
    print("正在加载数据...")
    # 加载数据
    user_positions, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale = load_data()
    print("数据加载完成！")
    
    print(f"用户总数: {NUM_USERS}")
    print(f"基站总数: {NUM_BS}")
    print(f"URLLC用户: U1-U10 (索引0-9)")
    print(f"eMBB用户: e1-e20 (索引10-29)")
    print(f"mMTC用户: m1-m40 (索引30-69)")
    print("-" * 50)
    
    # 结果采集容器
    times = []
    total_qos_hist = []
    urllc_avg_hist = []
    embb_avg_hist = []
    mmtc_avg_hist = []

    # 功率控制历史：shape (NUM_BS, T)
    power_history = []

    # 资源利用（按类型）用于柱状图（取最后一个时间点）
    last_time_resource_usage_by_type = None  # shape (NUM_BS, 3)

    # 接入分布统计（聚合整个时间窗口）
    access_counts_by_bs = np.zeros(NUM_BS, dtype=int)
    access_counts_by_bs_type = np.zeros((NUM_BS, 3), dtype=int)

    # SINR与速率分布（累计）
    sinr_samples_by_type = { 'URLLC': [], 'eMBB': [], 'mMTC': [] }
    rate_samples_by_type = { 'URLLC': [], 'eMBB': [], 'mMTC': [] }

    # 遗传算法收敛曲线（每个时间点一条）
    ga_convergence_histories = {}

    # 时延满足率累计
    sla_meet = { 'URLLC': [0, 0], 'eMBB': [0, 0], 'mMTC': [0, 0], 'OVERALL': [0, 0] }

    # 模拟一段时间内的资源配置 (例如1000ms内每100ms决策一次)
    for t in range(0, 1000, 100):
        time_index = t // 100
        print(f"\n时间: {t}ms")
        
        print("正在优化接入决策和资源分配...")
        # 优化接入决策和资源分配
        convergence_recorder = []
        best_individual = genetic_algorithm(time_index, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale, convergence_recorder)
        ga_convergence_histories[t] = convergence_recorder
        access_decision, resource_allocation = best_individual
        
        print("正在优化功率控制...")
        # 优化功率控制
        power_control = convex_optimization_power_control(access_decision, resource_allocation, time_index, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale, task_flow)
        
        print("正在计算用户服务质量...")
        # 计算用户服务质量
        sinr_values, rates = calculate_sinr_and_rate(access_decision, resource_allocation, power_control, time_index, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale)
        total_qos = 0
        urllc_qos = []
        embb_qos = []
        mmtc_qos = []

        # 本轮资源利用（按类型）统计
        res_usage_by_type = np.zeros((NUM_BS, 3), dtype=float)
        for u in range(NUM_USERS):
            bs = access_decision[u]
            allocated_blocks = resource_allocation[bs, u]
            if allocated_blocks == 0:
                continue
            user_type = USER_TYPES[u]
            try:
                time_row = task_flow.iloc[time_index]
                task_key = USER_COLUMN_MAP[u]
                queue_length = time_row.get(task_key, 1)
                # 考虑基站处理能力和用户任务类型
                bs_processing_factor = 1.0 if bs == 0 else 1.2  # MBS处理能力较强，SBS稍弱
                user_type_factor = 0.8 if user_type == 'URLLC' else (0.9 if user_type == 'eMBB' else 1.0)
                latency = queue_length * 0.1 * bs_processing_factor * user_type_factor  # 假设每个任务0.1ms，调整后的时延
            except:
                latency = 1.0
            qos = calculate_qos(user_type, rates[u], latency, allocated_blocks)
            total_qos += qos
            if user_type == 'URLLC':
                urllc_qos.append(qos)
            elif user_type == 'eMBB':
                embb_qos.append(qos)
            else:
                mmtc_qos.append(qos)

            # 累计SINR与速率样本（仅对有资源的用户）
            sinr_samples_by_type[user_type].append(10 * np.log10(max(sinr_values[u], 1e-12)))  # dB
            rate_samples_by_type[user_type].append(rates[u] / 1e6)  # Mbps

            # 资源使用（按类型）
            res_usage_by_type[bs, TYPE_TO_INDEX[user_type]] += allocated_blocks

            # SLA满足率统计（仅按时延阈值，符合题意）
            sla_meet['OVERALL'][1] += 1
            if user_type == 'URLLC':
                sla_meet['URLLC'][1] += 1
                if latency <= 1:
                    sla_meet['URLLC'][0] += 1
                    sla_meet['OVERALL'][0] += 1
            elif user_type == 'eMBB':
                sla_meet['eMBB'][1] += 1
                if latency <= 4:
                    sla_meet['eMBB'][0] += 1
                    sla_meet['OVERALL'][0] += 1
            else:  # mMTC
                sla_meet['mMTC'][1] += 1
                if latency <= 10:
                    sla_meet['mMTC'][0] += 1
                    sla_meet['OVERALL'][0] += 1
        
        urllc_avg_qos = np.mean(urllc_qos) if urllc_qos else 0
        embb_avg_qos = np.mean(embb_qos) if embb_qos else 0
        mmtc_avg_qos = np.mean(mmtc_qos) if mmtc_qos else 0
        
        # 保存结果到表格
        if t == 0:
            results_df = pd.DataFrame(columns=['time', 'total_qos', 'URLLC_avg_qos', 'eMBB_avg_qos', 'mMTC_avg_qos'])
        results_df = pd.concat([
            results_df,
            pd.DataFrame([
                {
                    'time': t,
                    'total_qos': total_qos,
                    'URLLC_avg_qos': urllc_avg_qos,
                    'eMBB_avg_qos': embb_avg_qos,
                    'mMTC_avg_qos': mmtc_avg_qos,
                }
            ])
        ], ignore_index=True)
        results_df.to_csv(RESULTS_PATH, index=False)
        
        print(f"总用户服务质量: {total_qos:.4f}")
        print(f"URLLC平均QoS: {urllc_avg_qos:.4f}")
        print(f"eMBB平均QoS: {embb_avg_qos:.4f}")
        print(f"mMTC平均QoS: {mmtc_avg_qos:.4f}")
        
        print(f"接入决策: {access_decision}")
        print(f"功率控制 (dBm): {power_control}")
        
        # 资源分配统计
        print("资源分配统计:")
        resource_usage = np.sum(resource_allocation, axis=1)
        print(f"  MBS_1: {resource_usage[0]}/100 资源块")
        print(f"  SBS_1: {resource_usage[1]}/50 资源块")
        print(f"  SBS_2: {resource_usage[2]}/50 资源块")
        print(f"  SBS_3: {resource_usage[3]}/50 资源块")
        print("-" * 50)

        # 采集时间序列数据
        times.append(t)
        total_qos_hist.append(total_qos)
        urllc_avg_hist.append(urllc_avg_qos)
        embb_avg_hist.append(embb_avg_qos)
        mmtc_avg_hist.append(mmtc_avg_qos)
        power_history.append(power_control)
        last_time_resource_usage_by_type = res_usage_by_type

        # 聚合接入分布
        for u in range(NUM_USERS):
            bs = access_decision[u]
            access_counts_by_bs[bs] += 1
            access_counts_by_bs_type[bs, TYPE_TO_INDEX[USER_TYPES[u]]] += 1

    # 绘图输出目录
    output_dir = os.path.dirname(RESULTS_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. 用户服务质量（QoS）随时间变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(times, total_qos_hist, label='总QoS值', linewidth=2)
    plt.plot(times, urllc_avg_hist, label='URLLC用户平均QoS', linewidth=2)
    plt.plot(times, embb_avg_hist, label='eMBB用户平均QoS', linewidth=2)
    plt.plot(times, mmtc_avg_hist, label='mMTC用户平均QoS', linewidth=2)
    plt.xlabel('时间 (ms)')
    plt.ylabel('QoS 值')
    plt.title('用户服务质量（QoS）随时间变化')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qos_over_time.png'), dpi=150)
    plt.close()

    # 2. 基站资源利用率堆叠柱状图（使用最后一个时间点）
    if last_time_resource_usage_by_type is not None:
        labels = ['MBS_1', 'SBS_1', 'SBS_2', 'SBS_3']
        urllc_vals = last_time_resource_usage_by_type[:, TYPE_TO_INDEX['URLLC']]
        embb_vals = last_time_resource_usage_by_type[:, TYPE_TO_INDEX['eMBB']]
        mmtc_vals = last_time_resource_usage_by_type[:, TYPE_TO_INDEX['mMTC']]
        x = np.arange(NUM_BS)
        width = 0.6
        plt.figure(figsize=(10, 6))
        p1 = plt.bar(x, urllc_vals, width, label='URLLC')
        p2 = plt.bar(x, embb_vals, width, bottom=urllc_vals, label='eMBB')
        p3 = plt.bar(x, mmtc_vals, width, bottom=urllc_vals + embb_vals, label='mMTC')
        # 参考线
        plt.hlines([100, 50, 50, 50], xmin=x-0.5, xmax=x+0.5, colors=['r','g','g','g'], linestyles='--',
                   label='资源上限 (MBS:100, SBS:50)')
        plt.xticks(x, labels)
        plt.ylabel('资源块使用量')
        plt.title('基站资源利用率（按类型堆叠）')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resource_usage_stacked.png'), dpi=150)
        plt.close()

    # 3. 功率控制热力图（基站 x 时间）
    if power_history:
        power_mat = np.array(power_history).T  # shape: (NUM_BS, T)
        plt.figure(figsize=(10, 4.5))
        im = plt.imshow(power_mat, aspect='auto', cmap='YlOrRd', origin='lower')
        plt.colorbar(im, label='功率 (dBm)')
        plt.yticks(np.arange(NUM_BS), ['MBS_1','SBS_1','SBS_2','SBS_3'])
        plt.xticks(np.arange(len(times)), [str(t) for t in times], rotation=45)
        plt.xlabel('时间 (ms)')
        plt.title('功率控制热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'power_heatmap.png'), dpi=150)
        plt.close()

    # 4. 用户接入分布嵌套饼图（聚合整个时间窗口）
    total_access = access_counts_by_bs.sum()
    if total_access > 0:
        outer_sizes = access_counts_by_bs / total_access
        bs_labels = ['MBS_1','SBS_1','SBS_2','SBS_3']
        # 内环：每个基站按类型比例
        inner_sizes = []
        inner_labels = []
        for bs in range(NUM_BS):
            bs_total = access_counts_by_bs[bs]
            type_counts = access_counts_by_bs_type[bs]
            if bs_total > 0:
                proportions = type_counts / bs_total
            else:
                proportions = np.zeros(3)
            inner_sizes.extend(proportions.tolist())
            inner_labels.extend([f"{bs_labels[bs]}-{INDEX_TO_TYPE[i]}" for i in range(3)])
        # 绘制
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(outer_sizes, radius=1.0, labels=bs_labels, autopct='%1.1f%%', pctdistance=0.85,
               wedgeprops=dict(width=0.3, edgecolor='w'))
        ax.pie(inner_sizes, radius=0.7, labels=None, autopct=None,
               wedgeprops=dict(width=0.3, edgecolor='w'))
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')
        ax.add_artist(centre_circle)
        ax.set(aspect="equal", title='用户接入分布（外环：基站比例；内环：各基站内类型比例）')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'access_distribution_donut.png'), dpi=150)
        plt.close()

    # 5. SINR与速率分布箱线图（双轴，按用户类型）
    categories = ['URLLC', 'eMBB', 'mMTC']
    sinr_data = [sinr_samples_by_type[c] if len(sinr_samples_by_type[c]) > 0 else [0] for c in categories]
    rate_data = [rate_samples_by_type[c] if len(rate_samples_by_type[c]) > 0 else [0] for c in categories]
    x_pos = np.arange(1, len(categories) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bp1 = ax1.boxplot(sinr_data, positions=x_pos, widths=0.35, patch_artist=True)
    for box in bp1['boxes']:
        box.set(facecolor='#87CEFA')
    ax1.set_ylabel('SINR (dB)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2 = ax1.twinx()
    bp2 = ax2.boxplot(rate_data, positions=x_pos + 0.4, widths=0.35, patch_artist=True)
    for box in bp2['boxes']:
        box.set(facecolor='#FFA07A')
    ax2.set_ylabel('传输速率 (Mbps)')
    ax1.set_title('SINR 与 速率分布（按用户类型）')
    ax1.legend([bp1['boxes'][0], bp2['boxes'][0]], ['SINR (dB)', '速率 (Mbps)'], loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sinr_rate_boxplot.png'), dpi=150)
    plt.close()

    # 6. 遗传算法收敛曲线（每个时间点一条）
    if ga_convergence_histories:
        plt.figure(figsize=(10, 6))
        for t, hist in ga_convergence_histories.items():
            if not hist:
                continue
            plt.plot(range(len(hist)), hist, label=f'{t}ms')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度（总QoS）')
        plt.title('遗传算法收敛曲线')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ga_convergence.png'), dpi=150)
        plt.close()

    # 7. 用户位置与基站覆盖散点图（使用最后一个时间点）
    try:
        # 选择最后一个时间点的用户位置
        pos_row = user_positions.iloc[times[-1] // 100]
        bs_coords = np.array([BS_POSITIONS[i] for i in range(NUM_BS)])
        plt.figure(figsize=(8, 8))
        plt.scatter(bs_coords[:, 0], bs_coords[:, 1], c=['red','orange','green','blue'], marker='^', s=200, label='基站')
        # 绘制用户点与连线
        colors = {'URLLC': 'red', 'eMBB': 'blue', 'mMTC': 'green'}
        # 需要再次获取该时刻的接入决策（简化：使用最后一次循环中的 access_decision）
        # 为确保变量存在，这里不访问局部变量，改为重算一次最优接入（不优化功率）
        temp_best_ind = genetic_algorithm(times[-1] // 100, task_flow, mbs_large_scale, mbs_small_scale, sbs_large_scale, sbs_small_scale)
        temp_access_decision, _ = temp_best_ind
        for u in range(NUM_USERS):
            user_type = USER_TYPES[u]
            name = USER_COLUMN_MAP[u]
            x = pos_row[f'{name}_X']
            y = pos_row[f'{name}_Y']
            plt.scatter([x], [y], c=colors[user_type], s=20)
            bs = temp_access_decision[u]
            bx, by = BS_POSITIONS[bs]
            plt.plot([x, bx], [y, by], color=colors[user_type], alpha=0.2, linewidth=0.8)
        plt.title('用户位置与基站覆盖（最后时刻）')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topology_scatter.png'), dpi=150)
        plt.close()
    except Exception:
        pass

    # 8. 时延满足率雷达图
    try:
        metrics = [
            ('URLLC时延≤1ms', sla_meet['URLLC']),
            ('eMBB速率≥50Mbps且时延≤4ms', sla_meet['eMBB']),
            ('mMTC时延≤10ms', sla_meet['mMTC']),
            ('总体SLA满足率', sla_meet['OVERALL'])
        ]
        values = []
        for _, (met, total) in metrics:
            pct = (met / total) if total > 0 else 0.0
            values.append(pct)
        # 闭合雷达
        labels = [m[0] for m in metrics]
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title('时延/总体SLA满足率雷达图')
        ax.set_rlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latency_sla_radar.png'), dpi=150)
        plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()