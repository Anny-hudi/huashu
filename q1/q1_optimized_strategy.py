import pandas as pd
import math
import numpy as np

def solve_problem_1_optimized_qos():
    """解决第一题：优化资源分配策略，在满足约束的前提下最大化QoS"""
    
    # 系统参数
    R_total = 50  # 总资源块数
    power = 30    # 发射功率 dBm
    bandwidth_per_rb = 360e3  # 360kHz
    thermal_noise = -174  # dBm/Hz
    NF = 7  # 噪声系数
    
    # SLA参数（根据body_and_more.md表1）
    URLLC_SLA_delay = 5    # ms
    eMBB_SLA_delay = 100   # ms
    mMTC_SLA_delay = 500   # ms
    URLLC_SLA_rate = 10    # Mbps
    eMBB_SLA_rate = 50     # Mbps
    mMTC_SLA_rate = 1      # Mbps
    
    # 惩罚系数（根据body_and_more.md表1）
    M_URLLC = 5
    M_eMBB = 3
    M_mMTC = 1
    alpha = 0.95  # URLLC效用折扣系数
    
    # 用户数量（根据data_1文件）
    URLLC_users = 2  # U1, U2
    eMBB_users = 4   # e1, e2, e3, e4
    mMTC_users = 10  # m1-m10
    
    # 每个用户的资源块占用量约束（根据body_and_more.md表1）
    URLLC_rb_per_user = 10  # 每个URLLC用户需要10个资源块
    eMBB_rb_per_user = 5    # 每个eMBB用户需要5个资源块
    mMTC_rb_per_user = 2    # 每个mMTC用户需要2个资源块
    
    def calculate_sinr(power_dbm, large_scale_db, small_scale, user_x, user_y, num_rbs):
        """计算信干噪比，使用大规模衰减、小规模瑞丽衰减和用户位置坐标"""
        power_mw = 10**((power_dbm - 30) / 10)
        
        # 基于用户位置坐标计算距离相关的路径损耗
        distance_m = math.sqrt(user_x**2 + user_y**2)  # 米
        distance_km = distance_m / 1000  # 转换为千米
        
        # 距离相关的路径损耗模型
        frequency_ghz = 2.6  # 假设使用2.6GHz频段
        distance_path_loss_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 147.55
        
        # 总信道增益 = 大规模衰减 + 小规模瑞丽衰减 + 距离路径损耗
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
        """计算URLLC服务质量 - 严格按照body_and_more.md定义"""
        if delay <= URLLC_SLA_delay:
            return alpha ** delay  # y^URLLC = α^L
        else:
            return -M_URLLC  # 延迟超时，给予惩罚
    
    def calculate_embb_qos(rate, delay):
        """计算eMBB服务质量 - 严格按照body_and_more.md定义"""
        if delay <= eMBB_SLA_delay:
            if rate >= eMBB_SLA_rate:
                return 1.0  # r ≥ r_SLA & L ≤ L_SLA
            else:
                return rate / eMBB_SLA_rate  # r < r_SLA & L ≤ L_SLA
        else:
            return -M_eMBB  # L > L_SLA，延迟超时
    
    def calculate_mmtc_qos(connection_ratio, delay):
        """计算mMTC服务质量 - 严格按照body_and_more.md定义"""
        if delay <= mMTC_SLA_delay:
            return connection_ratio  # Σc_i' / Σc_i
        else:
            return -M_mMTC  # L > L_SLA，延迟超时
    
    def analyze_user_channel_conditions(user_data):
        """分析用户信道条件，为优化分配提供依据"""
        print(f"\n=== 用户信道条件分析 ===")
        
        # URLLC用户分析
        print(f"\n--- URLLC用户信道分析 ---")
        urllc_analysis = []
        for i in range(URLLC_users):
            user_key = f'U{i+1}'
            if user_key in user_data['large_scale']:
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                user_x = user_data['user_pos'][f'{user_key}_X']
                user_y = user_data['user_pos'][f'{user_key}_Y']
                distance = math.sqrt(user_x**2 + user_y**2)
                
                # 计算信道质量
                channel_quality = large_scale + 10 * math.log10(small_scale)
                
                # 计算在标准分配下的性能
                sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, URLLC_rb_per_user)
                rate = calculate_rate(sinr, URLLC_rb_per_user)
                transmission_time = task_size / rate * 1000
                
                urllc_analysis.append({
                    'user_key': user_key,
                    'distance': distance,
                    'channel_quality': channel_quality,
                    'task_size': task_size,
                    'rate': rate,
                    'transmission_time': transmission_time,
                    'qos_potential': calculate_urllc_qos(rate, transmission_time)
                })
        
        # 按QoS潜力排序
        urllc_analysis.sort(key=lambda x: x['qos_potential'], reverse=True)
        print(f"{'排名':<4} {'用户':<6} {'距离(m)':<10} {'信道质量(dB)':<12} {'速率(Mbps)':<12} {'延迟(ms)':<10} {'QoS潜力':<10}")
        print(f"{'-'*4} {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for i, info in enumerate(urllc_analysis):
            print(f"{i+1:<4} {info['user_key']:<6} {info['distance']:<10.1f} {info['channel_quality']:<12.2f} {info['rate']:<12.2f} {info['transmission_time']:<10.4f} {info['qos_potential']:<10.4f}")
        
        # eMBB用户分析
        print(f"\n--- eMBB用户信道分析 ---")
        embb_analysis = []
        for i in range(eMBB_users):
            user_key = f'e{i+1}'
            if user_key in user_data['large_scale']:
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                user_x = user_data['user_pos'][f'{user_key}_X']
                user_y = user_data['user_pos'][f'{user_key}_Y']
                distance = math.sqrt(user_x**2 + user_y**2)
                
                # 计算信道质量
                channel_quality = large_scale + 10 * math.log10(small_scale)
                
                # 计算在标准分配下的性能
                sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, eMBB_rb_per_user)
                rate = calculate_rate(sinr, eMBB_rb_per_user)
                transmission_time = task_size / rate * 1000
                
                embb_analysis.append({
                    'user_key': user_key,
                    'distance': distance,
                    'channel_quality': channel_quality,
                    'task_size': task_size,
                    'rate': rate,
                    'transmission_time': transmission_time,
                    'qos_potential': calculate_embb_qos(rate, transmission_time),
                    'rate_deficit': max(0, eMBB_SLA_rate - rate)  # 速率缺口
                })
        
        # 按速率缺口排序（优先改善表现差的用户）
        embb_analysis.sort(key=lambda x: x['rate_deficit'], reverse=True)
        print(f"{'排名':<4} {'用户':<6} {'距离(m)':<10} {'信道质量(dB)':<12} {'速率(Mbps)':<12} {'延迟(ms)':<10} {'QoS潜力':<10} {'速率缺口':<10}")
        print(f"{'-'*4} {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for i, info in enumerate(embb_analysis):
            print(f"{i+1:<4} {info['user_key']:<6} {info['distance']:<10.1f} {info['channel_quality']:<12.2f} {info['rate']:<12.2f} {info['transmission_time']:<10.4f} {info['qos_potential']:<10.4f} {info['rate_deficit']:<10.2f}")
        
        return urllc_analysis, embb_analysis
    
    def optimize_resource_allocation(user_data):
        """优化资源分配策略"""
        print(f"\n=== 优化资源分配策略 ===")
        
        # 分析用户信道条件
        urllc_analysis, embb_analysis = analyze_user_channel_conditions(user_data)
        
        # 策略1: 优先改善eMBB表现差的用户
        print(f"\n--- 策略1: 优先改善eMBB表现差的用户 ---")
        
        # 识别需要额外资源的eMBB用户
        problematic_embb_users = []
        for info in embb_analysis:
            if info['qos_potential'] < 0:  # 有负QoS的用户
                problematic_embb_users.append(info)
        
        print(f"需要改善的eMBB用户: {[info['user_key'] for info in problematic_embb_users]}")
        
        # 计算需要额外分配的资源
        extra_embb_rbs = 0
        for info in problematic_embb_users:
            # 计算需要多少额外资源块才能达到SLA速率
            current_rate = info['rate']
            target_rate = eMBB_SLA_rate
            if current_rate < target_rate:
                # 估算需要的额外资源块
                rate_ratio = target_rate / current_rate
                extra_rbs_needed = math.ceil(eMBB_rb_per_user * (rate_ratio - 1))
                extra_embb_rbs += extra_rbs_needed
                print(f"  {info['user_key']}: 当前速率={current_rate:.2f}Mbps, 目标={target_rate}Mbps, 需要额外{extra_rbs_needed}RB")
        
        # 策略2: 优化mMTC分配
        print(f"\n--- 策略2: 优化mMTC分配 ---")
        
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
                
                # 计算在标准分配下的性能
                sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, mMTC_rb_per_user)
                rate = calculate_rate(sinr, mMTC_rb_per_user)
                connected = rate >= mMTC_SLA_rate
                
                mmtc_analysis.append({
                    'user_key': user_key,
                    'distance': distance,
                    'channel_quality': channel_quality,
                    'task_size': task_size,
                    'priority_score': priority_score,
                    'rate': rate,
                    'connected': connected
                })
        
        # 按优先级排序
        mmtc_analysis.sort(key=lambda x: x['priority_score'], reverse=True)
        print(f"{'排名':<4} {'用户':<6} {'优先级分数':<12} {'距离(m)':<10} {'信道质量(dB)':<12} {'速率(Mbps)':<12} {'连接状态':<8}")
        print(f"{'-'*4} {'-'*6} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
        for i, info in enumerate(mmtc_analysis):
            status = "✓" if info['connected'] else "✗"
            print(f"{i+1:<4} {info['user_key']:<6} {info['priority_score']:<12.6f} {info['distance']:<10.1f} {info['channel_quality']:<12.2f} {info['rate']:<12.2f} {status:<8}")
        
        # 策略3: 生成优化分配方案
        print(f"\n--- 策略3: 生成优化分配方案 ---")
        
        # 基础分配
        base_urllc_rbs = URLLC_users * URLLC_rb_per_user  # 20 RB
        base_embb_rbs = eMBB_users * eMBB_rb_per_user      # 20 RB
        base_mmtc_rbs = R_total - base_urllc_rbs - base_embb_rbs  # 10 RB
        
        print(f"基础分配: URLLC={base_urllc_rbs}RB, eMBB={base_embb_rbs}RB, mMTC={base_mmtc_rbs}RB")
        
        # 优化方案1: 给eMBB分配更多资源
        if extra_embb_rbs > 0:
            optimized_embb_rbs = base_embb_rbs + extra_embb_rbs
            optimized_mmtc_rbs = max(0, base_mmtc_rbs - extra_embb_rbs)
            
            print(f"优化方案1: URLLC={base_urllc_rbs}RB, eMBB={optimized_embb_rbs}RB, mMTC={optimized_mmtc_rbs}RB")
            print(f"  - 给eMBB增加{extra_embb_rbs}RB来改善表现差的用户")
            print(f"  - mMTC减少{extra_embb_rbs}RB，但通过优先级分配优化连接率")
        
        # 优化方案2: 基于用户优先级重新分配
        print(f"\n优化方案2: 基于用户优先级重新分配")
        
        # 计算每个用户的QoS改善潜力
        urllc_qos_improvement = []
        for info in urllc_analysis:
            # 计算增加资源块后的QoS改善
            current_rbs = URLLC_rb_per_user
            current_sinr = calculate_sinr(power, info['channel_quality'], 1, info['distance'], 0, current_rbs)
            current_rate = calculate_rate(current_sinr, current_rbs)
            current_qos = info['qos_potential']
            
            # 估算增加资源块后的改善
            improved_rbs = current_rbs + 5  # 增加5个资源块
            improved_sinr = calculate_sinr(power, info['channel_quality'], 1, info['distance'], 0, improved_rbs)
            improved_rate = calculate_rate(improved_sinr, improved_rbs)
            improved_transmission_time = info['task_size'] / improved_rate * 1000
            improved_qos = calculate_urllc_qos(improved_rate, improved_transmission_time)
            
            qos_improvement = improved_qos - current_qos
            urllc_qos_improvement.append({
                'user_key': info['user_key'],
                'current_qos': current_qos,
                'improved_qos': improved_qos,
                'improvement': qos_improvement
            })
        
        # 按QoS改善潜力排序
        urllc_qos_improvement.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"URLLC用户QoS改善潜力:")
        for info in urllc_qos_improvement:
            print(f"  {info['user_key']}: 当前QoS={info['current_qos']:.4f}, 改善后QoS={info['improved_qos']:.4f}, 改善={info['improvement']:.4f}")
        
        return {
            'base_allocation': (base_urllc_rbs, base_embb_rbs, base_mmtc_rbs),
            'optimized_allocation': (base_urllc_rbs, optimized_embb_rbs, optimized_mmtc_rbs) if extra_embb_rbs > 0 else None,
            'urllc_analysis': urllc_analysis,
            'embb_analysis': embb_analysis,
            'mmtc_analysis': mmtc_analysis,
            'problematic_embb_users': problematic_embb_users
        }
    
    # 加载data_1目录中的实际数据
    print("=== 加载data_1目录中的实际数据 ===")
    
    # 读取大规模衰减数据
    large_scale_data = pd.read_csv('data_1/大规模衰减.csv')
    large_scale = large_scale_data.iloc[0]
    
    # 读取小规模瑞丽衰减数据
    small_scale_data = pd.read_csv('data_1/小规模瑞丽衰减.csv')
    small_scale = small_scale_data.iloc[0]
    
    # 读取任务流数据
    task_flow_data = pd.read_csv('data_1/任务流.csv')
    task_flow = task_flow_data.iloc[0]
    
    # 读取用户位置数据
    user_pos_data = pd.read_csv('data_1/用户位置.csv')
    user_pos = user_pos_data.iloc[0]
    
    # 整合数据
    user_data = {
        'large_scale': large_scale,
        'small_scale': small_scale,
        'task_flow': task_flow,
        'user_pos': user_pos
    }
    
    print("=== 第一题：优化资源分配策略提升QoS ===")
    
    # 执行优化分析
    optimization_result = optimize_resource_allocation(user_data)
    
    print(f"\n=== 优化建议总结 ===")
    print(f"1. 识别出{len(optimization_result['problematic_embb_users'])}个需要改善的eMBB用户")
    print(f"2. 建议给eMBB分配额外资源来改善表现差的用户")
    print(f"3. 通过优先级分配优化mMTC连接率")
    print(f"4. 基于用户信道条件进行差异化资源分配")
    
    return optimization_result

if __name__ == "__main__":
    solve_problem_1_optimized_qos() 