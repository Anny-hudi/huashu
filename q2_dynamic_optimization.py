import pandas as pd
import math
import numpy as np
from collections import defaultdict, deque
import random

class DynamicResourceOptimizer:
    """动态资源优化器 - 基于MPC和强化学习"""
    
    def __init__(self):
        # 系统参数
        self.R_total = 50  # 总资源块数
        self.power = 30    # 发射功率 dBm
        self.bandwidth_per_rb = 360e3  # 360kHz
        self.thermal_noise = -174  # dBm/Hz
        self.NF = 7  # 噪声系数
        
        # SLA参数
        self.URLLC_SLA_delay = 5    # ms
        self.eMBB_SLA_delay = 100   # ms
        self.mMTC_SLA_delay = 500   # ms
        self.URLLC_SLA_rate = 10    # Mbps
        self.eMBB_SLA_rate = 50     # Mbps
        self.mMTC_SLA_rate = 1      # Mbps
        
        # 惩罚系数
        self.M_URLLC = 5
        self.M_eMBB = 3
        self.M_mMTC = 1
        self.alpha = 0.95  # URLLC效用折扣系数
        
        # 用户数量
        self.URLLC_users = 2  # U1, U2
        self.eMBB_users = 4   # e1, e2, e3, e4
        self.mMTC_users = 10  # m1-m10
        
        # 强化学习参数
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.epsilon = 0.1  # 探索率
        
        # MPC参数
        self.prediction_horizon = 3  # 预测时域长度
        self.control_horizon = 1     # 控制时域长度
        
        # 模拟退火参数
        self.initial_temperature = 100
        self.cooling_rate = 0.95
        self.min_temperature = 0.1
        
        # 用户映射
        self.user_mapping = {
            'URLLC': ['U1', 'U2'],
            'eMBB': ['e1', 'e2', 'e3', 'e4'],
            'mMTC': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
        }
        
        # 状态空间
        self.state_size = 16  # 用户数量
        self.action_size = 51  # 0-50个资源块
        
        # Q-learning表
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        
    def calculate_sinr(self, power_dbm, large_scale_db, small_scale, user_x, user_y, num_rbs):
        """计算信干噪比"""
        power_mw = 10**((power_dbm - 30) / 10)
        
        # 计算用户到基站的距离
        distance_m = math.sqrt(user_x**2 + user_y**2)
        distance_km = distance_m / 1000
        
        # 自由空间路径损耗模型
        frequency_ghz = 2.6
        distance_path_loss_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 147.55
        
        # 总信道增益
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
        """计算URLLC服务质量"""
        if delay <= self.URLLC_SLA_delay:
            return self.alpha ** delay
        else:
            return -self.M_URLLC
    
    def calculate_embb_qos(self, rate, delay):
        """计算eMBB服务质量"""
        if delay <= self.eMBB_SLA_delay:
            if rate >= self.eMBB_SLA_rate:
                return 1.0
            else:
                return rate / self.eMBB_SLA_rate
        else:
            return -self.M_eMBB
    
    def calculate_mmtc_qos(self, connection_ratio, delay):
        """计算mMTC服务质量"""
        if delay <= self.mMTC_SLA_delay:
            return connection_ratio
        else:
            return -self.M_mMTC
    
    def get_state_representation(self, user_data, queue_tasks):
        """获取状态表示"""
        state = []
        
        # 当前活跃任务数
        active_tasks = sum(1 for user, info in user_data.items() if info['task_size'] > 0)
        state.append(active_tasks / 16)  # 归一化
        
        # 队列长度
        queue_length = len(queue_tasks)
        state.append(min(queue_length / 100, 1.0))  # 归一化
        
        # 各切片任务比例
        urllc_tasks = sum(1 for user in self.user_mapping['URLLC'] 
                         if user in user_data and user_data[user]['task_size'] > 0)
        embb_tasks = sum(1 for user in self.user_mapping['eMBB'] 
                        if user in user_data and user_data[user]['task_size'] > 0)
        mmtc_tasks = sum(1 for user in self.user_mapping['mMTC'] 
                        if user in user_data and user_data[user]['task_size'] > 0)
        
        state.append(urllc_tasks / 2)   # 归一化
        state.append(embb_tasks / 4)    # 归一化
        state.append(mmtc_tasks / 10)   # 归一化
        
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
            state.append((avg_channel_quality + 100) / 200)  # 归一化到[0,1]
        else:
            state.append(0.5)
        
        return tuple(state)
    
    def mpc_optimization(self, user_data, queue_tasks, prediction_data):
        """模型预测控制优化"""
        best_allocation = None
        best_qos = -float('inf')
        
        # 预测未来几个时隙的任务到达
        future_demands = self.predict_future_demands(prediction_data)
        
        # 滚动时域优化
        for urllc_rbs in range(0, self.R_total + 1):
            for embb_rbs in range(0, self.R_total + 1):
                for mmtc_rbs in range(0, self.R_total + 1):
                    if urllc_rbs + embb_rbs + mmtc_rbs == self.R_total:
                        # 计算当前时隙QoS
                        current_qos = self.evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, 
                                                             user_data, queue_tasks)
                        
                        # 预测未来时隙QoS
                        future_qos = 0
                        for t in range(1, self.prediction_horizon + 1):
                            if t < len(future_demands):
                                future_qos += self.evaluate_future_allocation(urllc_rbs, embb_rbs, mmtc_rbs,
                                                                           future_demands[t]) * (self.discount_factor ** t)
                        
                        total_qos = current_qos + future_qos
                        
                        if total_qos > best_qos:
                            best_qos = total_qos
                            best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
        
        return best_allocation, best_qos
    
    def simulated_annealing_optimization(self, user_data, queue_tasks):
        """模拟退火优化"""
        # 初始解
        current_allocation = (20, 20, 10)  # URLLC, eMBB, mMTC
        current_qos = self.evaluate_allocation(*current_allocation, user_data, queue_tasks)
        
        best_allocation = current_allocation
        best_qos = current_qos
        
        temperature = self.initial_temperature
        
        while temperature > self.min_temperature:
            # 生成邻域解
            neighbor = self.generate_neighbor(current_allocation)
            neighbor_qos = self.evaluate_allocation(*neighbor, user_data, queue_tasks)
            
            # 计算接受概率
            delta_qos = neighbor_qos - current_qos
            if delta_qos > 0 or random.random() < math.exp(delta_qos / temperature):
                current_allocation = neighbor
                current_qos = neighbor_qos
                
                if current_qos > best_qos:
                    best_allocation = current_allocation
                    best_qos = current_qos
            
            temperature *= self.cooling_rate
        
        return best_allocation, best_qos
    
    def generate_neighbor(self, allocation):
        """生成邻域解"""
        urllc_rbs, embb_rbs, mmtc_rbs = allocation
        
        # 随机调整资源分配
        delta = random.randint(-5, 5)
        new_urllc = max(0, min(self.R_total, urllc_rbs + delta))
        new_embb = max(0, min(self.R_total, embb_rbs + random.randint(-3, 3)))
        new_mmtc = self.R_total - new_urllc - new_embb
        
        if new_mmtc < 0:
            new_mmtc = 0
            new_urllc = self.R_total - new_embb
        
        return (new_urllc, new_embb, new_mmtc)
    
    def q_learning_optimization(self, user_data, queue_tasks):
        """Q-learning优化"""
        state = self.get_state_representation(user_data, queue_tasks)
        
        # ε-贪婪策略
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            action = random.randint(0, self.action_size - 1)
        else:
            # 利用：选择Q值最大的动作
            action = np.argmax(self.q_table[state])
        
        # 执行动作（资源分配）
        allocation = self.action_to_allocation(action)
        qos = self.evaluate_allocation(*allocation, user_data, queue_tasks)
        
        # 更新Q表
        next_state = self.get_state_representation(user_data, queue_tasks)  # 简化
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + \
                                     self.learning_rate * (qos + self.discount_factor * max_next_q)
        
        return allocation, qos
    
    def action_to_allocation(self, action):
        """将动作转换为资源分配"""
        # 简单的映射：action 0-16 -> URLLC, 17-33 -> eMBB, 34-50 -> mMTC
        if action <= 16:
            return (action * 3, 0, self.R_total - action * 3)
        elif action <= 33:
            return (0, (action - 17) * 3, self.R_total - (action - 17) * 3)
        else:
            return (0, 0, self.R_total)
    
    def evaluate_allocation(self, urllc_rbs, embb_rbs, mmtc_rbs, user_data, queue_tasks):
        """评估资源分配方案的QoS"""
        total_qos = 0
        
        # 处理URLLC用户
        if urllc_rbs > 0:
            urllc_rb_per_user = urllc_rbs / self.URLLC_users
            for user in self.user_mapping['URLLC']:
                if user in user_data and user_data[user]['task_size'] > 0:
                    user_info = user_data[user]
                    sinr = self.calculate_sinr(self.power, user_info.get('large_scale', 0), 
                                             user_info.get('small_scale', 1), 
                                             user_info['x'], user_info['y'], urllc_rb_per_user)
                    rate = self.calculate_rate(sinr, urllc_rb_per_user)
                    delay = user_info['task_size'] / rate * 1000 if rate > 0 else float('inf')
                    qos = self.calculate_urllc_qos(rate, delay)
                    total_qos += qos
        
        # 处理eMBB用户
        if embb_rbs > 0:
            embb_rb_per_user = embb_rbs / self.eMBB_users
            for user in self.user_mapping['eMBB']:
                if user in user_data and user_data[user]['task_size'] > 0:
                    user_info = user_data[user]
                    sinr = self.calculate_sinr(self.power, user_info.get('large_scale', 0), 
                                             user_info.get('small_scale', 1), 
                                             user_info['x'], user_info['y'], embb_rb_per_user)
                    rate = self.calculate_rate(sinr, embb_rb_per_user)
                    delay = user_info['task_size'] / rate * 1000 if rate > 0 else float('inf')
                    qos = self.calculate_embb_qos(rate, delay)
                    total_qos += qos
        
        # 处理mMTC用户
        if mmtc_rbs > 0:
            mmtc_rb_per_user = mmtc_rbs / self.mMTC_users
            connected_users = 0
            total_users = len(self.user_mapping['mMTC'])
            
            for user in self.user_mapping['mMTC']:
                if user in user_data and user_data[user]['task_size'] > 0:
                    user_info = user_data[user]
                    sinr = self.calculate_sinr(self.power, user_info.get('large_scale', 0), 
                                             user_info.get('small_scale', 1), 
                                             user_info['x'], user_info['y'], mmtc_rb_per_user)
                    rate = self.calculate_rate(sinr, mmtc_rb_per_user)
                    
                    if rate >= self.mMTC_SLA_rate:
                        connected_users += 1
            
            connection_ratio = connected_users / total_users if total_users > 0 else 0
            delay = 100  # 简化延迟计算
            qos = self.calculate_mmtc_qos(connection_ratio, delay)
            total_qos += qos
        
        return total_qos
    
    def predict_future_demands(self, prediction_data):
        """预测未来需求"""
        # 简化的预测：基于历史数据趋势
        future_demands = {}
        for t in range(1, self.prediction_horizon + 1):
            future_demands[t] = prediction_data  # 简化处理
        return future_demands
    
    def evaluate_future_allocation(self, urllc_rbs, embb_rbs, mmtc_rbs, future_demand):
        """评估未来分配的QoS"""
        # 简化实现
        return 0.5  # 假设未来QoS为0.5
    
    def hybrid_optimization(self, user_data, queue_tasks, prediction_data):
        """混合优化策略"""
        # 1. MPC优化
        mpc_allocation, mpc_qos = self.mpc_optimization(user_data, queue_tasks, prediction_data)
        
        # 2. 模拟退火优化
        sa_allocation, sa_qos = self.simulated_annealing_optimization(user_data, queue_tasks)
        
        # 3. Q-learning优化
        ql_allocation, ql_qos = self.q_learning_optimization(user_data, queue_tasks)
        
        # 选择最佳结果
        results = [
            (mpc_allocation, mpc_qos, "MPC"),
            (sa_allocation, sa_qos, "Simulated Annealing"),
            (ql_allocation, ql_qos, "Q-Learning")
        ]
        
        best_result = max(results, key=lambda x: x[1])
        return best_result[0], best_result[1], best_result[2]

def solve_problem_2_dynamic():
    """使用动态优化和强化学习解决第二问"""
    
    # 初始化优化器
    optimizer = DynamicResourceOptimizer()
    
    # 时间参数
    total_time = 1000  # ms
    decision_interval = 100  # ms
    num_decisions = total_time // decision_interval  # 10次决策
    
    print(f"=== 第二问：动态优化和强化学习解决方案 ===")
    print(f"总时间: {total_time}ms, 决策间隔: {decision_interval}ms, 决策次数: {num_decisions}")
    
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
        """获取指定时间点的用户数据"""
        time_point = time_idx * decision_interval / 1000  # 转换为秒
        
        # 找到最接近的时间点
        time_diff = abs(task_flow_data['Time'] - time_point)
        closest_idx = time_diff.idxmin()
        
        user_data = {}
        
        # 获取任务到达数据
        for slice_type, users in optimizer.user_mapping.items():
            for user in users:
                task_size = task_flow_data.loc[closest_idx, user]
                user_data[user] = {
                    'slice_type': slice_type,
                    'task_size': task_size,
                    'time': time_point
                }
        
        # 获取用户位置数据
        for slice_type, users in optimizer.user_mapping.items():
            for user in users:
                x_col = f"{user}_X"
                y_col = f"{user}_Y"
                if x_col in user_position_data.columns and y_col in user_position_data.columns:
                    user_data[user]['x'] = user_position_data.loc[closest_idx, x_col]
                    user_data[user]['y'] = user_position_data.loc[closest_idx, y_col]
        
        # 获取信道数据
        for slice_type, users in optimizer.user_mapping.items():
            for user in users:
                if user in large_scale_data.columns:
                    user_data[user]['large_scale'] = large_scale_data.loc[closest_idx, user]
                if user in small_scale_data.columns:
                    user_data[user]['small_scale'] = small_scale_data.loc[closest_idx, user]
        
        return user_data
    
    # 主优化循环
    print(f"\n=== 开始动态优化 ===")
    
    # 任务队列
    task_queue = deque()
    all_allocations = []
    
    for decision_idx in range(num_decisions):
        print(f"\n--- 第 {decision_idx + 1} 次决策 (时间: {decision_idx * decision_interval}ms) ---")
        
        # 获取当前时间点的用户数据
        user_data = get_user_data_at_time(decision_idx)
        
        # 统计当前时间点的任务情况
        active_tasks = 0
        for user, info in user_data.items():
            if info['task_size'] > 0:
                active_tasks += 1
                task_queue.append({
                    'user': user,
                    'slice_type': info['slice_type'],
                    'task_size': info['task_size'],
                    'x': info['x'],
                    'y': info['y'],
                    'large_scale': info.get('large_scale', 0),
                    'small_scale': info.get('small_scale', 1),
                    'arrival_time': decision_idx * decision_interval
                })
        
        print(f"当前时间点活跃任务数: {active_tasks}")
        print(f"队列中任务数: {len(task_queue)}")
        
        # 预测数据（简化）
        prediction_data = user_data
        
        # 混合优化
        best_allocation, best_qos, method = optimizer.hybrid_optimization(user_data, task_queue, prediction_data)
        
        if best_allocation:
            urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
            print(f"最优分配 ({method}): URLLC={urllc_rbs}, eMBB={embb_rbs}, mMTC={mmtc_rbs}")
            print(f"总QoS: {best_qos:.4f}")
            
            all_allocations.append({
                'decision_idx': decision_idx,
                'time': decision_idx * decision_interval,
                'urllc_rbs': urllc_rbs,
                'embb_rbs': embb_rbs,
                'mmtc_rbs': mmtc_rbs,
                'total_qos': best_qos,
                'active_tasks': active_tasks,
                'method': method
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
                'method': 'None'
            })
        
        # 处理队列中的任务
        if len(task_queue) > 30:
            task_queue.popleft()
    
    # 输出结果
    print(f"\n=== 最终结果 ===")
    print("决策时间序列资源分配方案:")
    for allocation in all_allocations:
        print(f"时间 {allocation['time']}ms: URLLC={allocation['urllc_rbs']}, "
              f"eMBB={allocation['embb_rbs']}, mMTC={allocation['mmtc_rbs']}, "
              f"QoS={allocation['total_qos']:.4f}, 活跃任务={allocation['active_tasks']}, "
              f"方法={allocation['method']}")
    
    # 计算总体性能
    total_qos = sum(allocation['total_qos'] for allocation in all_allocations)
    avg_qos = total_qos / len(all_allocations)
    
    print(f"\n总体性能:")
    print(f"总QoS: {total_qos:.4f}")
    print(f"平均QoS: {avg_qos:.4f}")
    
    # 分析算法使用情况
    method_counts = {}
    for allocation in all_allocations:
        method = allocation['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\n算法使用统计:")
    for method, count in method_counts.items():
        print(f"{method}: {count}次")
    
    return all_allocations

if __name__ == "__main__":
    solve_problem_2_dynamic() 