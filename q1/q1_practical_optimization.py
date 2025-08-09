import pandas as pd
import math
import numpy as np

def solve_problem_1_practical_optimization():
    """解决第一题：实际可行的QoS优化方案"""
    
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
    alpha = 0.95
    
    # 用户数量
    URLLC_users = 2
    eMBB_users = 4
    mMTC_users = 10
    
    # 资源块约束
    URLLC_rb_per_user = 10
    eMBB_rb_per_user = 5
    mMTC_rb_per_user = 2
    
    def calculate_sinr(power_dbm, large_scale_db, small_scale, user_x, user_y, num_rbs):
        """计算信干噪比"""
        power_mw = 10**((power_dbm - 30) / 10)
        distance_m = math.sqrt(user_x**2 + user_y**2)
        distance_km = distance_m / 1000
        frequency_ghz = 2.6
        distance_path_loss_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 147.55
        total_channel_gain_db = large_scale_db + 10 * math.log10(small_scale) - distance_path_loss_db
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
    
    def evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data):
        """评估分配方案的总QoS"""
        total_qos = 0.0
        
        # URLLC评估
        urllc_qos_sum = 0.0
        for i in range(URLLC_users):
            user_key = f'U{i+1}'
            if user_key in user_data['large_scale']:
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                user_x = user_data['user_pos'][f'{user_key}_X']
                user_y = user_data['user_pos'][f'{user_key}_Y']
                
                sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, urllc_rbs)
                rate = calculate_rate(sinr, urllc_rbs)
                transmission_time = task_size / rate * 1000
                
                qos = calculate_urllc_qos(rate, transmission_time)
                urllc_qos_sum += qos
                total_qos += qos
        
        # eMBB评估
        embb_qos_sum = 0.0
        for i in range(eMBB_users):
            user_key = f'e{i+1}'
            if user_key in user_data['large_scale']:
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                user_x = user_data['user_pos'][f'{user_key}_X']
                user_y = user_data['user_pos'][f'{user_key}_Y']
                
                sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, embb_rbs)
                rate = calculate_rate(sinr, embb_rbs)
                transmission_time = task_size / rate * 1000
                
                qos = calculate_embb_qos(rate, transmission_time)
                embb_qos_sum += qos
                total_qos += qos
        
        # mMTC评估（优先级分配）
        mmtc_allocation = calculate_mmtc_priority_allocation(mmtc_rbs, user_data)
        connected_users = sum(1 for alloc in mmtc_allocation if alloc['rbs'] > 0)
        connection_ratio = connected_users / mMTC_users if mMTC_users > 0 else 0
        
        # 计算平均延迟
        total_task_size = sum(user_data['task_flow'][f'm{i+1}'] for i in range(mMTC_users))
        avg_task_size = total_task_size / mMTC_users
        total_allocated_rbs = sum(alloc['rbs'] for alloc in mmtc_allocation)
        if total_allocated_rbs > 0:
            avg_rate = total_allocated_rbs * bandwidth_per_rb * math.log2(1 + 1) / 1e6
            avg_delay = avg_task_size / avg_rate * 1000
        else:
            avg_delay = float('inf')
        
        mmtc_qos = calculate_mmtc_qos(connection_ratio, avg_delay)
        total_qos += mmtc_qos
        
        return total_qos, urllc_qos_sum, embb_qos_sum, mmtc_qos
    
    def calculate_mmtc_priority_allocation(mmtc_rbs, user_data):
        """计算mMTC用户的优先级分配方案"""
        if mmtc_rbs == 0:
            return []
        
        # 计算可以满足完整约束的用户数量
        full_constraint_users = mmtc_rbs // mMTC_rb_per_user
        remaining_rbs = mmtc_rbs % mMTC_rb_per_user
        
        # 为mMTC用户计算优先级分数
        user_scores = []
        for i in range(mMTC_users):
            user_key = f'm{i+1}'
            if user_key in user_data['large_scale']:
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                user_x = user_data['user_pos'][f'{user_key}_X']
                user_y = user_data['user_pos'][f'{user_key}_Y']
                distance = math.sqrt(user_x**2 + user_y**2)
                
                # 计算优先级分数
                channel_quality = large_scale + 10 * math.log10(small_scale)
                priority_score = (1.0 / distance) * channel_quality * task_size
                
                user_scores.append({
                    'user_id': i,
                    'user_key': user_key,
                    'priority_score': priority_score,
                    'distance': distance,
                    'channel_quality': channel_quality,
                    'task_size': task_size
                })
        
        # 按优先级排序
        user_scores.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 分配资源块
        allocation = []
        for i, user_info in enumerate(user_scores):
            if i < full_constraint_users:
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': mMTC_rb_per_user,
                    'priority_rank': i + 1,
                    'constraint_satisfied': True
                })
            elif i < full_constraint_users + remaining_rbs:
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': 1,
                    'priority_rank': i + 1,
                    'constraint_satisfied': False
                })
            else:
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': 0,
                    'priority_rank': i + 1,
                    'constraint_satisfied': False
                })
        
        allocation.sort(key=lambda x: x['user_id'])
        return allocation
    
    def find_optimal_allocation(user_data):
        """寻找最优分配方案"""
        print(f"\n=== 寻找最优分配方案 ===")
        
        # 策略1: 基础分配（当前最优）
        base_allocation = (20, 20, 10)  # URLLC=20, eMBB=20, mMTC=10
        base_qos, base_urllc, base_embb, base_mmtc = evaluate_allocation(*base_allocation, user_data)
        
        print(f"基础分配方案: URLLC=20, eMBB=20, mMTC=10")
        print(f"总QoS: {base_qos:.4f}")
        print(f"URLLC QoS: {base_urllc:.4f}")
        print(f"eMBB QoS: {base_embb:.4f}")
        print(f"mMTC QoS: {base_mmtc:.4f}")
        
        # 策略2: 优化eMBB分配
        print(f"\n--- 策略2: 优化eMBB分配 ---")
        
        # 分析eMBB用户的问题
        embb_analysis = []
        for i in range(eMBB_users):
            user_key = f'e{i+1}'
            if user_key in user_data['large_scale']:
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                user_x = user_data['user_pos'][f'{user_key}_X']
                user_y = user_data['user_pos'][f'{user_key}_Y']
                
                # 当前分配下的性能
                sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, 5)
                rate = calculate_rate(sinr, 5)
                transmission_time = task_size / rate * 1000
                qos = calculate_embb_qos(rate, transmission_time)
                
                embb_analysis.append({
                    'user_key': user_key,
                    'current_rate': rate,
                    'current_qos': qos,
                    'rate_deficit': max(0, eMBB_SLA_rate - rate)
                })
        
        # 找出问题最严重的用户
        problematic_users = [info for info in embb_analysis if info['current_qos'] < 0]
        print(f"问题用户: {[info['user_key'] for info in problematic_users]}")
        
        # 策略3: 尝试不同的分配组合
        print(f"\n--- 策略3: 尝试不同分配组合 ---")
        
        best_qos = base_qos
        best_allocation = base_allocation
        
        # 尝试给eMBB分配更多资源
        for extra_embb in range(5, 16, 5):  # 尝试增加5, 10, 15个资源块
            urllc_rbs = 20
            embb_rbs = 20 + extra_embb
            mmtc_rbs = 10 - extra_embb
            
            if mmtc_rbs >= 0:  # 确保mMTC还有资源
                qos, urllc_qos, embb_qos, mmtc_qos = evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
                
                print(f"分配方案: URLLC={urllc_rbs}, eMBB={embb_rbs}, mMTC={mmtc_rbs}")
                print(f"  总QoS: {qos:.4f} (URLLC: {urllc_qos:.4f}, eMBB: {embb_qos:.4f}, mMTC: {mmtc_qos:.4f})")
                
                if qos > best_qos:
                    best_qos = qos
                    best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
                    print(f"  *** 发现更好的方案！ ***")
        
        # 策略4: 优化mMTC分配
        print(f"\n--- 策略4: 优化mMTC分配 ---")
        
        # 分析mMTC用户的优先级
        mmtc_analysis = []
        for i in range(mMTC_users):
            user_key = f'm{i+1}'
            if user_key in user_data['large_scale']:
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                user_x = user_data['user_pos'][f'{user_key}_X']
                user_y = user_data['user_pos'][f'{user_key}_Y']
                distance = math.sqrt(user_x**2 + user_y**2)
                
                # 计算优先级分数
                channel_quality = large_scale + 10 * math.log10(small_scale)
                priority_score = (1.0 / distance) * channel_quality * task_size
                
                # 计算在2RB分配下的性能
                sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, 2)
                rate = calculate_rate(sinr, 2)
                connected = rate >= mMTC_SLA_rate
                
                mmtc_analysis.append({
                    'user_key': user_key,
                    'priority_score': priority_score,
                    'rate': rate,
                    'connected': connected,
                    'distance': distance
                })
        
        # 按优先级排序
        mmtc_analysis.sort(key=lambda x: x['priority_score'], reverse=True)
        print(f"mMTC用户优先级排名:")
        for i, info in enumerate(mmtc_analysis):
            status = "✓" if info['connected'] else "✗"
            print(f"  {i+1}. {info['user_key']}: 优先级={info['priority_score']:.6f}, 速率={info['rate']:.2f}Mbps, 连接={status}")
        
        return best_allocation, best_qos
    
    # 加载数据
    print("=== 加载data_1目录中的实际数据 ===")
    
    large_scale_data = pd.read_csv('data_1/大规模衰减.csv')
    large_scale = large_scale_data.iloc[0]
    
    small_scale_data = pd.read_csv('data_1/小规模瑞丽衰减.csv')
    small_scale = small_scale_data.iloc[0]
    
    task_flow_data = pd.read_csv('data_1/任务流.csv')
    task_flow = task_flow_data.iloc[0]
    
    user_pos_data = pd.read_csv('data_1/用户位置.csv')
    user_pos = user_pos_data.iloc[0]
    
    user_data = {
        'large_scale': large_scale,
        'small_scale': small_scale,
        'task_flow': task_flow,
        'user_pos': user_pos
    }
    
    print("=== 第一题：实际可行的QoS优化方案 ===")
    
    # 寻找最优分配
    optimal_allocation, optimal_qos = find_optimal_allocation(user_data)
    
    print(f"\n=== 最优分配方案 ===")
    print(f"URLLC: {optimal_allocation[0]} RB")
    print(f"eMBB: {optimal_allocation[1]} RB")
    print(f"mMTC: {optimal_allocation[2]} RB")
    print(f"总QoS: {optimal_qos:.4f}")
    
    # 详细分析最优方案
    print(f"\n=== 最优方案详细分析 ===")
    qos, urllc_qos, embb_qos, mmtc_qos = evaluate_allocation(*optimal_allocation, user_data)
    
    print(f"URLLC QoS: {urllc_qos:.4f}")
    print(f"eMBB QoS: {embb_qos:.4f}")
    print(f"mMTC QoS: {mmtc_qos:.4f}")
    
    return optimal_allocation, optimal_qos

if __name__ == "__main__":
    solve_problem_1_practical_optimization() 