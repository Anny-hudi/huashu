import pandas as pd
import math
import numpy as np
from collections import defaultdict, deque

def solve_problem_2_final():
    """解决第二题：最终版本，采用更灵活的约束策略"""
    
    # 系统参数
    R_total = 50  # 总资源块数
    power = 30    # 发射功率 dBm
    bandwidth_per_rb = 360e3  # 360kHz
    thermal_noise = -174  # dBm/Hz
    NF = 7  # 噪声系数
    
    # SLA参数
    URLLC_SLA_delay = 5    # ms
    eMBB_SLA_delay = 100   # ms
    mMTC_SLA_delay = 500   # ms
    URLLC_SLA_rate = 10    # Mbps
    eMBB_SLA_rate = 50     # Mbps
    mMTC_SLA_rate = 1      # Mbps
    
    # 惩罚系数
    M_URLLC = 5
    M_eMBB = 3
    M_mMTC = 1
    alpha = 0.95  # URLLC效用折扣系数
    
    # 用户数量
    URLLC_users = 2  # U1, U2
    eMBB_users = 4   # e1, e2, e3, e4
    mMTC_users = 10  # m1-m10
    
    # 每个用户的资源块占用量约束
    URLLC_rb_per_user = 10  # 每个URLLC用户需要10个资源块
    eMBB_rb_per_user = 5    # 每个eMBB用户需要5个资源块
    mMTC_rb_per_user = 2    # 每个mMTC用户需要2个资源块
    
    # 时间参数
    total_time = 1000  # ms
    decision_interval = 100  # ms
    num_decisions = total_time // decision_interval  # 10次决策
    
    print(f"=== 第二问：最终时间序列资源分配优化 ===")
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
    
    # 定义用户映射
    user_mapping = {
        'URLLC': ['U1', 'U2'],
        'eMBB': ['e1', 'e2', 'e3', 'e4'],
        'mMTC': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
    }
    
    def calculate_sinr(power_dbm, large_scale_db, small_scale, user_x, user_y, num_rbs):
        """计算信干噪比"""
        power_mw = 10**((power_dbm - 30) / 10)
        
        # 计算用户到基站的距离
        distance_m = math.sqrt(user_x**2 + user_y**2)
        distance_km = distance_m / 1000
        
        # 自由空间路径损耗模型
        frequency_ghz = 2.6
        distance_path_loss_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 147.55
        
        # 总信道增益
        # 确保small_scale为正值，避免log10(负数)错误
        small_scale_positive = max(small_scale, 0.001)  # 避免log10(0)或负数
        total_channel_gain_db = large_scale_db + 10 * math.log10(small_scale_positive) - distance_path_loss_db
        channel_gain_linear = 10**(total_channel_gain_db / 10)
        received_power = power_mw * channel_gain_linear
        
        noise_power = 10**((thermal_noise + 10*math.log10(num_rbs * bandwidth_per_rb) + NF) / 10)
        sinr = received_power / noise_power
        return sinr
    
    def calculate_rate(sinr, num_rbs):
        """计算传输速率 (Mbps)"""
        rate = num_rbs * bandwidth_per_rb * math.log2(1 + sinr)
        return rate / 1e6
    
    def calculate_urllc_qos(rate, delay):
        """计算URLLC服务质量"""
        if delay <= URLLC_SLA_delay:
            return alpha ** delay
        else:
            return -M_URLLC
    
    def calculate_embb_qos(rate, delay):
        """计算eMBB服务质量"""
        if delay <= eMBB_SLA_delay:
            if rate >= eMBB_SLA_rate:
                return 1.0
            else:
                return rate / eMBB_SLA_rate
        else:
            return -M_eMBB
    
    def calculate_mmtc_qos(connection_ratio, delay):
        """计算mMTC服务质量"""
        if delay <= mMTC_SLA_delay:
            return connection_ratio
        else:
            return -M_mMTC
    
    def get_user_data_at_time(time_idx):
        """获取指定时间点的用户数据"""
        time_point = time_idx * decision_interval / 1000  # 转换为秒
        
        # 找到最接近的时间点
        time_diff = abs(task_flow_data['Time'] - time_point)
        closest_idx = time_diff.idxmin()
        
        user_data = {}
        
        # 获取任务到达数据
        for slice_type, users in user_mapping.items():
            for user in users:
                task_size = task_flow_data.loc[closest_idx, user]
                user_data[user] = {
                    'slice_type': slice_type,
                    'task_size': task_size,
                    'time': time_point
                }
        
        # 获取用户位置数据
        for slice_type, users in user_mapping.items():
            for user in users:
                x_col = f"{user}_X"
                y_col = f"{user}_Y"
                if x_col in user_position_data.columns and y_col in user_position_data.columns:
                    user_data[user]['x'] = user_position_data.loc[closest_idx, x_col]
                    user_data[user]['y'] = user_position_data.loc[closest_idx, y_col]
        
        # 获取信道数据
        for slice_type, users in user_mapping.items():
            for user in users:
                if user in large_scale_data.columns:
                    user_data[user]['large_scale'] = large_scale_data.loc[closest_idx, user]
                if user in small_scale_data.columns:
                    user_data[user]['small_scale'] = small_scale_data.loc[closest_idx, user]
        
        return user_data
    
    def optimize_resource_allocation_flexible(user_data, queue_tasks):
        """采用更灵活的约束策略优化资源分配"""
        best_qos = -float('inf')
        best_allocation = None
        
        # 穷举搜索所有可能的资源分配
        for urllc_rbs in range(0, R_total + 1):
            for embb_rbs in range(0, R_total + 1):
                for mmtc_rbs in range(0, R_total + 1):
                    if urllc_rbs + embb_rbs + mmtc_rbs == R_total:
                        # 计算QoS
                        total_qos = 0
                        
                        # 处理URLLC用户
                        if urllc_rbs > 0:
                            urllc_rb_per_user_actual = urllc_rbs / URLLC_users
                            for user in user_mapping['URLLC']:
                                if user in user_data and user_data[user]['task_size'] > 0:
                                    user_info = user_data[user]
                                    
                                    # 计算SINR和速率
                                    sinr = calculate_sinr(power, user_info.get('large_scale', 0), 
                                                         user_info.get('small_scale', 1), 
                                                         user_info['x'], user_info['y'], urllc_rb_per_user_actual)
                                    rate = calculate_rate(sinr, urllc_rb_per_user_actual)
                                    
                                    # 计算延迟（简化模型）
                                    delay = user_info['task_size'] / rate * 1000 if rate > 0 else float('inf')
                                    
                                    qos = calculate_urllc_qos(rate, delay)
                                    total_qos += qos
                        
                        # 处理eMBB用户
                        if embb_rbs > 0:
                            embb_rb_per_user_actual = embb_rbs / eMBB_users
                            for user in user_mapping['eMBB']:
                                if user in user_data and user_data[user]['task_size'] > 0:
                                    user_info = user_data[user]
                                    
                                    sinr = calculate_sinr(power, user_info.get('large_scale', 0), 
                                                         user_info.get('small_scale', 1), 
                                                         user_info['x'], user_info['y'], embb_rb_per_user_actual)
                                    rate = calculate_rate(sinr, embb_rb_per_user_actual)
                                    
                                    delay = user_info['task_size'] / rate * 1000 if rate > 0 else float('inf')
                                    
                                    qos = calculate_embb_qos(rate, delay)
                                    total_qos += qos
                        
                        # 处理mMTC用户
                        if mmtc_rbs > 0:
                            mmtc_rb_per_user_actual = mmtc_rbs / mMTC_users
                            connected_users = 0
                            total_users = len(user_mapping['mMTC'])
                            
                            for user in user_mapping['mMTC']:
                                if user in user_data and user_data[user]['task_size'] > 0:
                                    user_info = user_data[user]
                                    
                                    sinr = calculate_sinr(power, user_info.get('large_scale', 0), 
                                                         user_info.get('small_scale', 1), 
                                                         user_info['x'], user_info['y'], mmtc_rb_per_user_actual)
                                    rate = calculate_rate(sinr, mmtc_rb_per_user_actual)
                                    
                                    if rate >= mMTC_SLA_rate:
                                        connected_users += 1
                            
                            connection_ratio = connected_users / total_users if total_users > 0 else 0
                            delay = 100  # 简化延迟计算
                            qos = calculate_mmtc_qos(connection_ratio, delay)
                            total_qos += qos
                        
                        # 处理排队任务
                        for task in queue_tasks:
                            if task['slice_type'] == 'URLLC' and urllc_rbs > 0:
                                urllc_rb_per_user_actual = urllc_rbs / URLLC_users
                                sinr = calculate_sinr(power, task.get('large_scale', 0), 
                                                     task.get('small_scale', 1), 
                                                     task['x'], task['y'], urllc_rb_per_user_actual)
                                rate = calculate_rate(sinr, urllc_rb_per_user_actual)
                                delay = task['task_size'] / rate * 1000 if rate > 0 else float('inf')
                                qos = calculate_urllc_qos(rate, delay)
                                total_qos += qos
                            
                            elif task['slice_type'] == 'eMBB' and embb_rbs > 0:
                                embb_rb_per_user_actual = embb_rbs / eMBB_users
                                sinr = calculate_sinr(power, task.get('large_scale', 0), 
                                                     task.get('small_scale', 1), 
                                                     task['x'], task['y'], embb_rb_per_user_actual)
                                rate = calculate_rate(sinr, embb_rb_per_user_actual)
                                delay = task['task_size'] / rate * 1000 if rate > 0 else float('inf')
                                qos = calculate_embb_qos(rate, delay)
                                total_qos += qos
                            
                            elif task['slice_type'] == 'mMTC' and mmtc_rbs > 0:
                                mmtc_rb_per_user_actual = mmtc_rbs / mMTC_users
                                sinr = calculate_sinr(power, task.get('large_scale', 0), 
                                                     task.get('small_scale', 1), 
                                                     task['x'], task['y'], mmtc_rb_per_user_actual)
                                rate = calculate_rate(sinr, mmtc_rb_per_user_actual)
                                if rate >= mMTC_SLA_rate:
                                    qos = calculate_mmtc_qos(1.0, 100)  # 简化
                                    total_qos += qos
                        
                        if total_qos > best_qos:
                            best_qos = total_qos
                            best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
        
        return best_allocation, best_qos
    
    # 主优化循环
    print(f"\n=== 开始时间序列优化 ===")
    
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
        
        # 优化资源分配
        best_allocation, best_qos = optimize_resource_allocation_flexible(user_data, task_queue)
        
        if best_allocation:
            urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
            print(f"最优分配: URLLC={urllc_rbs}, eMBB={embb_rbs}, mMTC={mmtc_rbs}")
            print(f"总QoS: {best_qos:.4f}")
            
            all_allocations.append({
                'decision_idx': decision_idx,
                'time': decision_idx * decision_interval,
                'urllc_rbs': urllc_rbs,
                'embb_rbs': embb_rbs,
                'mmtc_rbs': mmtc_rbs,
                'total_qos': best_qos,
                'active_tasks': active_tasks
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
                'active_tasks': active_tasks
            })
        
        # 处理队列中的任务（简化：移除已处理的任务）
        # 实际应用中需要更复杂的队列管理
        if len(task_queue) > 30:  # 限制队列长度
            task_queue.popleft()
    
    # 输出结果
    print(f"\n=== 最终结果 ===")
    print("决策时间序列资源分配方案:")
    for allocation in all_allocations:
        print(f"时间 {allocation['time']}ms: URLLC={allocation['urllc_rbs']}, "
              f"eMBB={allocation['embb_rbs']}, mMTC={allocation['mmtc_rbs']}, "
              f"QoS={allocation['total_qos']:.4f}, 活跃任务={allocation['active_tasks']}")
    
    # 计算总体性能
    total_qos = sum(allocation['total_qos'] for allocation in all_allocations)
    avg_qos = total_qos / len(all_allocations)
    
    print(f"\n总体性能:")
    print(f"总QoS: {total_qos:.4f}")
    print(f"平均QoS: {avg_qos:.4f}")
    
    # 分析资源分配模式
    urllc_total = sum(allocation['urllc_rbs'] for allocation in all_allocations)
    embb_total = sum(allocation['embb_rbs'] for allocation in all_allocations)
    mmtc_total = sum(allocation['mmtc_rbs'] for allocation in all_allocations)
    
    print(f"\n资源分配统计:")
    print(f"URLLC总资源: {urllc_total} RB")
    print(f"eMBB总资源: {embb_total} RB")
    print(f"mMTC总资源: {mmtc_total} RB")
    print(f"总分配资源: {urllc_total + embb_total + mmtc_total} RB")
    
    # 分析QoS分布
    positive_qos_count = sum(1 for allocation in all_allocations if allocation['total_qos'] > 0)
    negative_qos_count = sum(1 for allocation in all_allocations if allocation['total_qos'] < 0)
    
    print(f"\nQoS分布分析:")
    print(f"正QoS决策次数: {positive_qos_count}")
    print(f"负QoS决策次数: {negative_qos_count}")
    
    return all_allocations

if __name__ == "__main__":
    solve_problem_2_final() 