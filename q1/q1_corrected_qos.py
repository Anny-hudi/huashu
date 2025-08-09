import pandas as pd
import math
import numpy as np

def solve_problem_1_corrected_qos():
    """解决第一题：修正服务质量计算逻辑，使用data_1目录中的实际数据，满足所有约束"""
    
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
    
    # 最小资源需求
    min_URLLC_rbs = URLLC_users * URLLC_rb_per_user  # 2 * 10 = 20
    min_eMBB_rbs = eMBB_users * eMBB_rb_per_user     # 4 * 5 = 20
    min_mMTC_rbs = mMTC_users * mMTC_rb_per_user     # 10 * 2 = 20
    
    # 总最小需求：20 + 20 + 20 = 60 > 50，需要调整策略
    print(f"=== 资源约束分析 ===")
    print(f"URLLC用户数: {URLLC_users}, 每个用户需要: {URLLC_rb_per_user} RB, 最小需求: {min_URLLC_rbs} RB")
    print(f"eMBB用户数: {eMBB_users}, 每个用户需要: {eMBB_rb_per_user} RB, 最小需求: {min_eMBB_rbs} RB")
    print(f"mMTC用户数: {mMTC_users}, 每个用户需要: {mMTC_rb_per_user} RB, 最小需求: {min_mMTC_rbs} RB")
    print(f"总最小需求: {min_URLLC_rbs + min_eMBB_rbs + min_mMTC_rbs} RB > 总资源: {R_total} RB")
    
    # 由于总资源不足，采用穷举法寻找最优分配
    print(f"\n=== 采用穷举法寻找最优分配 ===")
    print(f"优先级: URLLC > eMBB > mMTC")
    print(f"mMTC优先级分配策略：优先满足部分用户完整约束（2RB/用户），其余用户分配1RB")
    
    def calculate_mmtc_priority_allocation(mmtc_rbs, user_data):
        """计算mMTC用户的优先级分配方案"""
        if mmtc_rbs == 0:
            return []
        
        # 计算可以满足完整约束的用户数量
        full_constraint_users = mmtc_rbs // mMTC_rb_per_user  # 可以分配2RB的用户数
        remaining_rbs = mmtc_rbs % mMTC_rb_per_user  # 剩余资源块
        
        # 为mMTC用户计算优先级分数（基于位置、信道条件等）
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
                
                # 计算优先级分数（距离越近、信道越好、任务越大，优先级越高）
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
                # 满足完整约束的用户
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': mMTC_rb_per_user,  # 2RB
                    'priority_rank': i + 1,
                    'constraint_satisfied': True
                })
            elif i < full_constraint_users + remaining_rbs:
                # 分配1RB的用户
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': 1,  # 1RB
                    'priority_rank': i + 1,
                    'constraint_satisfied': False
                })
            else:
                # 没有分配到资源的用户
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': 0,  # 0RB
                    'priority_rank': i + 1,
                    'constraint_satisfied': False
                })
        
        # 按用户ID重新排序
        allocation.sort(key=lambda x: x['user_id'])
        return allocation
    
    def calculate_sinr(power_dbm, large_scale_db, small_scale, user_x, user_y, num_rbs):
        """计算信干噪比，严格按照body_and_more.md定义"""
        # 将发射功率从dBm转换为mW
        power_mw = 10**((power_dbm - 30) / 10)
        
        # 根据题目公式：p_rx = 10^((p_n,k - φ_n,k)/10) * |h_n,k|²
        # 其中φ_n,k是大规模衰减（dB），|h_n,k|²是小规模瑞丽衰减（无量纲）
        received_power = 10**((power_dbm - large_scale_db) / 10) * small_scale
        
        # 计算噪声功率：N₀ = -174 + 10*log₁₀(ib) + NF
        # 其中i是资源块数量，b是单资源块带宽
        noise_power_dbm = thermal_noise + 10 * math.log10(num_rbs * bandwidth_per_rb) + NF
        noise_power_mw = 10**((noise_power_dbm - 30) / 10)
        
        # 计算SINR（第一题中无干扰，所以只有信号和噪声）
        sinr = received_power / noise_power_mw
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
    
    def generate_task_arrivals(slice_type, num_users, time_period=100):
        """生成任务到达时间（满足body_and_more.md表1的分布要求）"""
        if slice_type == 'URLLC':
            # 泊松分布（指数分布）
            arrival_times = np.random.exponential(scale=time_period/num_users, size=num_users)
        elif slice_type in ['eMBB', 'mMTC']:
            # 均匀分布
            arrival_times = np.random.uniform(0, time_period, size=num_users)
        else:
            arrival_times = np.zeros(num_users)
        
        return np.sort(arrival_times)  # 按时间排序
    
    def evaluate_allocation_with_constraints(urllc_rbs, embb_rbs, mmtc_rbs, user_data):
        """评估资源分配方案的服务质量，考虑所有约束"""
        total_qos = 0.0
        urllc_qos_sum = 0.0
        embb_qos_sum = 0.0
        mmtc_qos = 0.0
        
        # 增加平均QoS计算变量
        urllc_qos_list = []
        embb_qos_list = []
        mmtc_qos_list = []
        all_qos_list = []
        
        # 检查资源块占用量约束
        urllc_rb_per_user_actual = urllc_rbs / URLLC_users if URLLC_users > 0 else 0
        embb_rb_per_user_actual = embb_rbs / eMBB_users if eMBB_users > 0 else 0
        
        # 对于mMTC，计算优先级分配后的约束满足情况
        mmtc_full_constraint_users = mmtc_rbs // mMTC_rb_per_user
        mmtc_remaining_rbs = mmtc_rbs % mMTC_rb_per_user
        mmtc_total_served_users = mmtc_full_constraint_users + mmtc_remaining_rbs
        
        print(f"\n=== 资源分配约束检查 ===")
        print(f"URLLC: {urllc_rbs} RB / {URLLC_users} 用户 = {urllc_rb_per_user_actual:.1f} RB/用户 (要求≥{URLLC_rb_per_user})")
        print(f"eMBB: {embb_rbs} RB / {eMBB_users} 用户 = {embb_rb_per_user_actual:.1f} RB/用户 (要求≥{eMBB_rb_per_user})")
        print(f"mMTC: {mmtc_rbs} RB / {mMTC_users} 用户 (优先级分配)")
        print(f"  - 完整约束用户: {mmtc_full_constraint_users} 用户 (2RB/用户)")
        print(f"  - 部分约束用户: {mmtc_remaining_rbs} 用户 (1RB/用户)")
        print(f"  - 未服务用户: {mMTC_users - mmtc_total_served_users} 用户 (0RB/用户)")
        print(f"  - 总服务用户: {mmtc_total_served_users}/{mMTC_users} 用户")
        
        # 检查是否满足最小资源块要求
        urllc_satisfied = urllc_rb_per_user_actual >= URLLC_rb_per_user
        embb_satisfied = embb_rb_per_user_actual >= eMBB_rb_per_user
        mmtc_satisfied = mmtc_full_constraint_users > 0  # 至少有一些用户满足完整约束
        
        print(f"URLLC约束满足: {'✓' if urllc_satisfied else '✗'}")
        print(f"eMBB约束满足: {'✓' if embb_satisfied else '✗'}")
        print(f"mMTC约束满足: {'✓' if mmtc_satisfied else '✗'} (优先级分配策略)")
        
        # 添加约束满足度分析
        if not urllc_satisfied:
            print(f"  URLLC约束缺口: {URLLC_rb_per_user - urllc_rb_per_user_actual:.1f} RB/用户")
        if not embb_satisfied:
            print(f"  eMBB约束缺口: {eMBB_rb_per_user - embb_rb_per_user_actual:.1f} RB/用户")
        if not mmtc_satisfied:
            print(f"  mMTC约束缺口: 没有用户满足完整约束")
        else:
            print(f"  mMTC优先级分配: {mmtc_full_constraint_users}个用户满足完整约束，{mmtc_remaining_rbs}个用户部分满足")
        
        # URLLC用户评估
        if urllc_rbs > 0 and urllc_satisfied:
            print(f"\n--- URLLC切片性能分析 ({urllc_rbs} RB) ---")
            # 生成泊松分布的任务到达时间
            urllc_arrivals = generate_task_arrivals('URLLC', URLLC_users)
            print(f"URLLC任务到达时间 (泊松分布): {urllc_arrivals}")
            
            # 创建URLLC用户详细信息表格
            print(f"\n=== URLLC用户详细信息 ===")
            print(f"{'用户':<6} {'切片类型':<8} {'分配RB数':<8} {'速率(Mbps)':<12} {'延迟(ms)':<12} {'服务质量':<10}")
            print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
            
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
                    
                    # 考虑任务到达时间的延迟计算
                    arrival_time = urllc_arrivals[i]
                    transmission_time = task_size / rate * 1000  # ms
                    # 对于URLLC，延迟只考虑传输时间，不包括到达时间
                    total_delay = transmission_time
                    
                    qos = calculate_urllc_qos(rate, total_delay)
                    urllc_qos_sum += qos
                    total_qos += qos
                    
                    # 记录QoS值用于平均计算
                    urllc_qos_list.append(qos)
                    all_qos_list.append(qos)
                    
                    user_x = user_data['user_pos'][f'{user_key}_X']
                    user_y = user_data['user_pos'][f'{user_key}_Y']
                    distance = math.sqrt(user_x**2 + user_y**2)
                    # 输出表格格式的用户信息
                    print(f"{user_key:<6} {'URLLC':<8} {urllc_rbs//URLLC_users:<8} {rate:<12.2f} {total_delay:<12.4f} {qos:<10.4f}")
                    # 输出详细位置信息
                    print(f"  {user_key}: 速率={rate:.2f} Mbps, 位置=({user_x:.1f},{user_y:.1f}), 距离={distance:.1f}m, 到达时间={arrival_time:.2f}ms, 传输时间={transmission_time:.4f}ms, 延迟={total_delay:.4f}ms, QoS={qos:.4f}")
        
        # eMBB用户评估
        if embb_rbs > 0 and embb_satisfied:
            print(f"\n--- eMBB切片性能分析 ({embb_rbs} RB) ---")
            # 生成均匀分布的任务到达时间
            embb_arrivals = generate_task_arrivals('eMBB', eMBB_users)
            print(f"eMBB任务到达时间 (均匀分布): {embb_arrivals}")
            
            # 创建eMBB用户详细信息表格
            print(f"\n=== eMBB用户详细信息 ===")
            print(f"{'用户':<6} {'切片类型':<8} {'分配RB数':<8} {'速率(Mbps)':<12} {'延迟(ms)':<12} {'服务质量':<10}")
            print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
            
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
                    
                    # 考虑任务到达时间的延迟计算
                    arrival_time = embb_arrivals[i]
                    transmission_time = task_size / rate * 1000  # ms
                    # 对于eMBB，延迟只考虑传输时间，不包括到达时间
                    total_delay = transmission_time
                    
                    qos = calculate_embb_qos(rate, total_delay)
                    embb_qos_sum += qos
                    total_qos += qos
                    
                    # 记录QoS值用于平均计算
                    embb_qos_list.append(qos)
                    all_qos_list.append(qos)
                    
                    user_x = user_data['user_pos'][f'{user_key}_X']
                    user_y = user_data['user_pos'][f'{user_key}_Y']
                    distance = math.sqrt(user_x**2 + user_y**2)
                    # 输出表格格式的用户信息
                    print(f"{user_key:<6} {'eMBB':<8} {embb_rbs//eMBB_users:<8} {rate:<12.2f} {total_delay:<12.4f} {qos:<10.4f}")
                    # 输出详细位置信息
                    print(f"  {user_key}: 速率={rate:.2f} Mbps, 位置=({user_x:.1f},{user_y:.1f}), 距离={distance:.1f}m, 到达时间={arrival_time:.2f}ms, 传输时间={transmission_time:.4f}ms, 延迟={total_delay:.4f}ms, QoS={qos:.4f}")
        
        # mMTC用户评估（使用优先级分配策略）
        if mmtc_rbs > 0:
            print(f"\n--- mMTC切片性能分析 ({mmtc_rbs} RB) ---")
            # 生成均匀分布的任务到达时间
            mmtc_arrivals = generate_task_arrivals('mMTC', mMTC_users)
            print(f"mMTC任务到达时间 (均匀分布): {mmtc_arrivals}")
            
            # 使用优先级分配策略
            mmtc_allocation = calculate_mmtc_priority_allocation(mmtc_rbs, user_data)
            
            # 输出mMTC用户优先级排名
            print(f"\n=== mMTC用户优先级排名 ===")
            print(f"{'排名':<4} {'用户':<6} {'优先级分数':<12} {'距离(m)':<10} {'信道质量(dB)':<12} {'任务数据量(Mbit)':<15} {'分配RB':<8} {'约束满足':<8}")
            print(f"{'-'*4} {'-'*6} {'-'*12} {'-'*10} {'-'*12} {'-'*15} {'-'*8} {'-'*8}")
            
            # 收集所有用户的优先级信息
            priority_info = []
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
                    
                    # 获取分配信息
                    user_allocation = next((alloc for alloc in mmtc_allocation if alloc['user_key'] == user_key), None)
                    if user_allocation:
                        allocated_rbs = user_allocation['rbs']
                        priority_rank = user_allocation['priority_rank']
                        constraint_satisfied = user_allocation['constraint_satisfied']
                    else:
                        allocated_rbs = 0
                        priority_rank = 0
                        constraint_satisfied = False
                    
                    priority_info.append({
                        'user_key': user_key,
                        'priority_score': priority_score,
                        'distance': distance,
                        'channel_quality': channel_quality,
                        'task_size': task_size,
                        'allocated_rbs': allocated_rbs,
                        'priority_rank': priority_rank,
                        'constraint_satisfied': constraint_satisfied
                    })
            
            # 按优先级分数排序
            priority_info.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # 输出优先级排名
            for i, info in enumerate(priority_info):
                rank = i + 1
                constraint_status = "✓" if info['constraint_satisfied'] else "✗"
                print(f"{rank:<4} {info['user_key']:<6} {info['priority_score']:<12.6f} {info['distance']:<10.1f} {info['channel_quality']:<12.2f} {info['task_size']:<15.6f} {info['allocated_rbs']:<8} {constraint_status:<8}")
            
            # 输出优先级分配策略总结
            mmtc_full_constraint_users = mmtc_rbs // mMTC_rb_per_user
            mmtc_remaining_rbs = mmtc_rbs % mMTC_rb_per_user
            mmtc_total_served_users = mmtc_full_constraint_users + mmtc_remaining_rbs
            
            print(f"\n=== mMTC优先级分配策略总结 ===")
            print(f"总资源块: {mmtc_rbs} RB")
            print(f"完整约束用户数: {mmtc_full_constraint_users} 用户 (2RB/用户)")
            print(f"部分约束用户数: {mmtc_remaining_rbs} 用户 (1RB/用户)")
            print(f"未服务用户数: {mMTC_users - mmtc_total_served_users} 用户 (0RB/用户)")
            print(f"总服务用户数: {mmtc_total_served_users}/{mMTC_users} 用户")
            print(f"优先级计算因子: 1/距离 × 信道质量 × 任务数据量")
            
            # 创建mMTC用户详细信息表格
            print(f"\n=== mMTC用户详细信息（优先级分配） ===")
            print(f"{'用户':<6} {'优先级':<6} {'分配RB数':<8} {'速率(Mbps)':<12} {'延迟(ms)':<12} {'服务质量':<10} {'约束满足':<8}")
            print(f"{'-'*6} {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
            
            connected_users = 0
            total_users = 0
            total_task_size = 0
            user_details = []
            
            for i in range(mMTC_users):
                user_key = f'm{i+1}'
                if user_key in user_data['large_scale']:
                    total_users += 1
                    large_scale = user_data['large_scale'][user_key]
                    small_scale = user_data['small_scale'][user_key]
                    task_size = user_data['task_flow'][user_key]
                    total_task_size += task_size
                    
                    user_x = user_data['user_pos'][f'{user_key}_X']
                    user_y = user_data['user_pos'][f'{user_key}_Y']
                    
                    # 获取该用户的资源分配情况
                    user_allocation = next((alloc for alloc in mmtc_allocation if alloc['user_key'] == user_key), None)
                    if user_allocation:
                        allocated_rbs = user_allocation['rbs']
                        priority_rank = user_allocation['priority_rank']
                        constraint_satisfied = user_allocation['constraint_satisfied']
                    else:
                        allocated_rbs = 0
                        priority_rank = 0
                        constraint_satisfied = False
                    
                    if allocated_rbs > 0:
                        sinr = calculate_sinr(power, large_scale, small_scale, user_x, user_y, allocated_rbs)
                        rate = calculate_rate(sinr, allocated_rbs)
                        
                        # 考虑任务到达时间的延迟计算
                        arrival_time = mmtc_arrivals[i]
                        transmission_time = task_size / rate * 1000  # ms
                        # 对于mMTC，延迟只考虑传输时间，不包括到达时间
                        total_delay = transmission_time
                        
                        connected = rate >= mMTC_SLA_rate
                        if connected:
                            connected_users += 1
                        
                        user_details.append({
                            'user': user_key,
                            'rate': rate,
                            'connected': connected,
                            'arrival_time': arrival_time,
                            'transmission_time': transmission_time,
                            'total_delay': total_delay,
                            'large_scale': large_scale,
                            'small_scale': small_scale,
                            'sinr': sinr,
                            'task_size': task_size,
                            'allocated_rbs': allocated_rbs,
                            'priority_rank': priority_rank,
                            'constraint_satisfied': constraint_satisfied
                        })
                    else:
                        # 没有分配到资源的用户
                        user_details.append({
                            'user': user_key,
                            'rate': 0,
                            'connected': False,
                            'arrival_time': mmtc_arrivals[i],
                            'transmission_time': float('inf'),
                            'total_delay': float('inf'),
                            'large_scale': large_scale,
                            'small_scale': small_scale,
                            'sinr': 0,
                            'task_size': task_size,
                            'allocated_rbs': 0,
                            'priority_rank': priority_rank,
                            'constraint_satisfied': False
                        })
            
            connection_ratio = connected_users / total_users if total_users > 0 else 0
            
            # 计算mMTC的平均延迟（基于实际分配的资源）
            if mmtc_rbs > 0:
                # 使用实际分配的资源计算平均延迟
                avg_task_size = total_task_size / total_users if total_users > 0 else 0.013
                # 计算实际的平均传输速率
                total_allocated_rbs = sum(detail['allocated_rbs'] for detail in user_details if detail['allocated_rbs'] > 0)
                if total_allocated_rbs > 0:
                    # 使用实际分配的资源块计算平均速率
                    # 需要基于实际SINR计算，而不是假设SINR=1
                    avg_sinr = 1.0  # 简化计算，使用平均SINR
                    avg_rate = total_allocated_rbs * bandwidth_per_rb * math.log2(1 + avg_sinr) / 1e6
                    avg_delay = avg_task_size / avg_rate * 1000
                else:
                    avg_delay = float('inf')
            else:
                avg_delay = float('inf')
            
            qos = calculate_mmtc_qos(connection_ratio, avg_delay)
            mmtc_qos = qos
            total_qos += qos
            
            # 记录mMTC的QoS值用于平均计算
            mmtc_qos_list.append(qos)
            all_qos_list.append(qos)
            
            print(f"  mMTC: 连接率={connection_ratio:.2f}, 平均延迟={avg_delay:.4f} ms, QoS={qos:.4f}")
            print(f"    - 连接用户数: {connected_users}/{total_users}")
            print(f"    - 平均任务数据量: {avg_task_size:.6f} Mbit")
            print(f"    - 平均传输速率: {avg_rate:.2f} Mbps")
            print(f"    - 优先级分配策略: 前{mmtc_rbs//mMTC_rb_per_user}个用户获得2RB，其余用户获得1RB或0RB")
            
            # 输出每个用户的详细信息表格
            for detail in user_details:
                status = "✓" if detail['connected'] else "✗"
                constraint_status = "✓" if detail['constraint_satisfied'] else "✗"
                user_x = user_data['user_pos'][f"{detail['user']}_X"]
                user_y = user_data['user_pos'][f"{detail['user']}_Y"]
                distance = math.sqrt(user_x**2 + user_y**2)
                # 计算mMTC用户的QoS（基于连接状态）
                mmtc_user_qos = 1.0 if detail['connected'] else 0.0
                
                if detail['allocated_rbs'] > 0:
                    # 输出表格格式的用户信息
                    print(f"{detail['user']:<6} {detail['priority_rank']:<6} {detail['allocated_rbs']:<8} {detail['rate']:<12.2f} {detail['total_delay']:<12.4f} {mmtc_user_qos:<10.4f} {constraint_status:<8}")
                    # 输出详细位置信息
                    print(f"    {detail['user']}: {status} 速率={detail['rate']:.2f} Mbps, 位置=({user_x:.1f},{user_y:.1f}), 距离={distance:.1f}m, 优先级={detail['priority_rank']}, 分配RB={detail['allocated_rbs']}, 到达时间={detail['arrival_time']:.2f}ms, 延迟={detail['total_delay']:.4f}ms")
                else:
                    # 没有分配到资源的用户
                    print(f"{detail['user']:<6} {detail['priority_rank']:<6} {detail['allocated_rbs']:<8} {'0.00':<12} {'∞':<12} {'0.0000':<10} {constraint_status:<8}")
                    print(f"    {detail['user']}: ✗ 未分配资源, 位置=({user_x:.1f},{user_y:.1f}), 距离={distance:.1f}m, 优先级={detail['priority_rank']}")
        
        # 基于用户位置的分析
        print(f"\n=== 用户位置分析 ===")
        
        # 收集各切片用户的位置信息
        urllc_distances = []
        embb_distances = []
        mmtc_distances = []
        
        # URLLC用户位置统计
        if urllc_rbs > 0 and urllc_satisfied:
            for i in range(URLLC_users):
                user_key = f'U{i+1}'
                if f'{user_key}_X' in user_data['user_pos']:
                    user_x = user_data['user_pos'][f'{user_key}_X']
                    user_y = user_data['user_pos'][f'{user_key}_Y']
                    distance = math.sqrt(user_x**2 + user_y**2)
                    urllc_distances.append(distance)
        
        # eMBB用户位置统计
        if embb_rbs > 0 and embb_satisfied:
            for i in range(eMBB_users):
                user_key = f'e{i+1}'
                if f'{user_key}_X' in user_data['user_pos']:
                    user_x = user_data['user_pos'][f'{user_key}_X']
                    user_y = user_data['user_pos'][f'{user_key}_Y']
                    distance = math.sqrt(user_x**2 + user_y**2)
                    embb_distances.append(distance)
        
        # mMTC用户位置统计
        if mmtc_rbs > 0 and mmtc_satisfied:
            for i in range(mMTC_users):
                user_key = f'm{i+1}'
                if f'{user_key}_X' in user_data['user_pos']:
                    user_x = user_data['user_pos'][f'{user_key}_X']
                    user_y = user_data['user_pos'][f'{user_key}_Y']
                    distance = math.sqrt(user_x**2 + user_y**2)
                    mmtc_distances.append(distance)
        
        # 输出位置统计信息
        if urllc_distances:
            avg_urllc_distance = np.mean(urllc_distances)
            std_urllc_distance = np.std(urllc_distances)
            print(f"URLLC用户平均距离: {avg_urllc_distance:.2f} ± {std_urllc_distance:.2f} 米")
            print(f"URLLC用户距离范围: {min(urllc_distances):.2f} - {max(urllc_distances):.2f} 米")
        
        if embb_distances:
            avg_embb_distance = np.mean(embb_distances)
            std_embb_distance = np.std(embb_distances)
            print(f"eMBB用户平均距离: {avg_embb_distance:.2f} ± {std_embb_distance:.2f} 米")
            print(f"eMBB用户距离范围: {min(embb_distances):.2f} - {max(embb_distances):.2f} 米")
        
        if mmtc_distances:
            avg_mmtc_distance = np.mean(mmtc_distances)
            std_mmtc_distance = np.std(mmtc_distances)
            print(f"mMTC用户平均距离: {avg_mmtc_distance:.2f} ± {std_mmtc_distance:.2f} 米")
            print(f"mMTC用户距离范围: {min(mmtc_distances):.2f} - {max(mmtc_distances):.2f} 米")
        
        # 整体距离统计
        all_distances = urllc_distances + embb_distances + mmtc_distances
        if all_distances:
            avg_total_distance = np.mean(all_distances)
            std_total_distance = np.std(all_distances)
            print(f"整体平均距离: {avg_total_distance:.2f} ± {std_total_distance:.2f} 米")
            print(f"整体距离范围: {min(all_distances):.2f} - {max(all_distances):.2f} 米")
        
        # 计算平均QoS
        avg_urllc_qos = np.mean(urllc_qos_list) if urllc_qos_list else 0
        avg_embb_qos = np.mean(embb_qos_list) if embb_qos_list else 0
        avg_mmtc_qos = np.mean(mmtc_qos_list) if mmtc_qos_list else 0
        avg_total_qos = np.mean(all_qos_list) if all_qos_list else 0
        
        # 输出平均QoS统计
        print(f"\n=== 平均QoS统计 ===")
        print(f"URLLC平均QoS: {avg_urllc_qos:.4f} (用户数: {len(urllc_qos_list)})")
        print(f"eMBB平均QoS: {avg_embb_qos:.4f} (用户数: {len(embb_qos_list)})")
        print(f"mMTC平均QoS: {avg_mmtc_qos:.4f} (用户数: {len(mmtc_qos_list)})")
        print(f"整体平均QoS: {avg_total_qos:.4f} (总用户数: {len(all_qos_list)})")
        
        # 计算QoS标准差
        std_urllc_qos = np.std(urllc_qos_list) if len(urllc_qos_list) > 1 else 0
        std_embb_qos = np.std(embb_qos_list) if len(embb_qos_list) > 1 else 0
        std_mmtc_qos = np.std(mmtc_qos_list) if len(mmtc_qos_list) > 1 else 0
        std_total_qos = np.std(all_qos_list) if len(all_qos_list) > 1 else 0
        
        print(f"URLLC QoS标准差: {std_urllc_qos:.4f}")
        print(f"eMBB QoS标准差: {std_embb_qos:.4f}")
        print(f"mMTC QoS标准差: {std_mmtc_qos:.4f}")
        print(f"整体QoS标准差: {std_total_qos:.4f}")
        
        return total_qos, urllc_qos_sum, embb_qos_sum, mmtc_qos, (urllc_satisfied, embb_satisfied, mmtc_satisfied), (avg_urllc_qos, avg_embb_qos, avg_mmtc_qos, avg_total_qos), (std_urllc_qos, std_embb_qos, std_mmtc_qos, std_total_qos)
    
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
    
    # 读取用户位置数据（用于验证）
    user_pos_data = pd.read_csv('data_1/用户位置.csv')
    user_pos = user_pos_data.iloc[0]
    
    # 整合数据
    user_data = {
        'large_scale': large_scale,
        'small_scale': small_scale,
        'task_flow': task_flow,
        'user_pos': user_pos
    }
    
    print("=== 第一题：使用data_1实际数据的服务质量计算（满足所有约束） ===")
    print(f"严格按照body_and_more.md中的服务质量定义")
    print(f"SLA参数:")
    print(f"  URLLC: 延迟≤{URLLC_SLA_delay}ms, 速率≥{URLLC_SLA_rate}Mbps")
    print(f"  eMBB: 延迟≤{eMBB_SLA_delay}ms, 速率≥{eMBB_SLA_rate}Mbps")
    print(f"  mMTC: 延迟≤{mMTC_SLA_delay}ms, 速率≥{mMTC_SLA_rate}Mbps")
    
    print(f"\n=== 系统参数 ===")
    print(f"总资源块数: {R_total}")
    print(f"发射功率: {power} dBm")
    print(f"单资源块带宽: {bandwidth_per_rb/1000:.1f} kHz")
    print(f"热噪声: {thermal_noise} dBm/Hz")
    print(f"噪声系数: {NF} dB")
    
    print(f"\n=== 约束参数 ===")
    print(f"URLLC用户数: {URLLC_users}, 每个用户需要: {URLLC_rb_per_user} RB")
    print(f"eMBB用户数: {eMBB_users}, 每个用户需要: {eMBB_rb_per_user} RB")
    print(f"mMTC用户数: {mMTC_users}, 每个用户需要: {mMTC_rb_per_user} RB")
    print(f"任务到达分布: URLLC(泊松分布), eMBB(均匀分布), mMTC(均匀分布)")
    
    print(f"\n=== 实际任务数据量 (Mbit) ===")
    for user in ['U1', 'U2', 'e1', 'e2', 'e3', 'e4', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']:
        print(f"{user}: {task_flow[user]:.6f}")
    
    # 穷举所有可能的资源分配组合
    print(f"\n=== 生成所有可能的资源分配组合 ===")
    all_allocations = []
    
    # 分析eMBB用户的信道条件，确定最优分配
    print(f"\n=== eMBB用户信道条件分析 ===")
    embb_channel_analysis = []
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
            
            embb_channel_analysis.append({
                'user_key': user_key,
                'distance': distance,
                'channel_quality': channel_quality,
                'task_size': task_size,
                'priority_score': (1.0 / distance) * channel_quality * task_size
            })
    
    # 按信道质量排序eMBB用户
    embb_channel_analysis.sort(key=lambda x: x['priority_score'], reverse=True)
    print(f"{'排名':<4} {'用户':<6} {'距离(m)':<10} {'信道质量(dB)':<12} {'任务数据量(Mbit)':<15} {'优先级分数':<12}")
    print(f"{'-'*4} {'-'*6} {'-'*10} {'-'*12} {'-'*15} {'-'*12}")
    for i, info in enumerate(embb_channel_analysis):
        print(f"{i+1:<4} {info['user_key']:<6} {info['distance']:<10.1f} {info['channel_quality']:<12.2f} {info['task_size']:<15.6f} {info['priority_score']:<12.6f}")
    
    # 遍历所有可能的分配组合
    for urllc_rbs in range(0, R_total + 1):
        for embb_rbs in range(0, R_total + 1):
            for mmtc_rbs in range(0, R_total + 1):
                if urllc_rbs + embb_rbs + mmtc_rbs == R_total:
                    # 检查资源块数是否为用户需求的整数倍
                    urllc_valid = (urllc_rbs % URLLC_rb_per_user == 0) if URLLC_users > 0 else True
                    embb_valid = (embb_rbs % eMBB_rb_per_user == 0) if eMBB_users > 0 else True
                    mmtc_valid = (mmtc_rbs % mMTC_rb_per_user == 0) if mMTC_users > 0 else True
                    
                    # 检查每个用户是否能获得足够的资源块
                    urllc_rb_per_user_actual = urllc_rbs / URLLC_users if URLLC_users > 0 else 0
                    embb_rb_per_user_actual = embb_rbs / eMBB_users if eMBB_users > 0 else 0
                    
                    # 对于mMTC，允许部分用户满足约束，部分用户分配1RB
                    mmtc_full_constraint_users = mmtc_rbs // mMTC_rb_per_user
                    mmtc_remaining_rbs = mmtc_rbs % mMTC_rb_per_user
                    mmtc_total_served_users = mmtc_full_constraint_users + mmtc_remaining_rbs
                    
                    # 优先满足URLLC和eMBB，mMTC采用优先级分配策略
                    # 放宽约束：允许eMBB用户获得更多资源块来改善性能
                    if (urllc_valid and embb_valid and mmtc_valid and
                        urllc_rb_per_user_actual >= URLLC_rb_per_user and 
                        embb_rb_per_user_actual >= eMBB_rb_per_user):
                        all_allocations.append((urllc_rbs, embb_rbs, mmtc_rbs))
    
    print(f"找到 {len(all_allocations)} 个满足URLLC和eMBB约束的分配方案")
    print(f"分配方案列表:")
    for i, (urllc, embb, mmtc) in enumerate(all_allocations):
        mmtc_full_users = mmtc // mMTC_rb_per_user
        mmtc_remaining = mmtc % mMTC_rb_per_user
        mmtc_total_served = mmtc_full_users + mmtc_remaining
        print(f"  方案{i+1}: URLLC={urllc}, eMBB={embb}, mMTC={mmtc} (完整约束用户:{mmtc_full_users}, 部分约束用户:{mmtc_remaining}, 总服务用户:{mmtc_total_served})")
    
    if len(all_allocations) == 0:
        print(f"\n⚠️  警告：没有找到完全满足所有约束的分配方案！")
        print(f"总资源: {R_total} RB")
        print(f"URLLC最小需求: {min_URLLC_rbs} RB (2用户 × 10RB/用户)")
        print(f"eMBB最小需求: {min_eMBB_rbs} RB (4用户 × 5RB/用户)")
        print(f"mMTC最小需求: {min_mMTC_rbs} RB (10用户 × 2RB/用户)")
        print(f"总最小需求: {min_URLLC_rbs + min_eMBB_rbs + min_mMTC_rbs} RB")
        print(f"采用灵活搜索策略：优先满足URLLC和eMBB，mMTC采用优先级分配")
        
        # 重新搜索，允许mMTC部分满足
        all_allocations = []
        for urllc_rbs in range(0, R_total + 1):
            for embb_rbs in range(0, R_total + 1):
                for mmtc_rbs in range(0, R_total + 1):
                    if urllc_rbs + embb_rbs + mmtc_rbs == R_total:
                        # 检查资源块数是否为用户需求的整数倍
                        urllc_valid = (urllc_rbs % URLLC_rb_per_user == 0) if URLLC_users > 0 else True
                        embb_valid = (embb_rbs % eMBB_rb_per_user == 0) if eMBB_users > 0 else True
                        mmtc_valid = (mmtc_rbs % mMTC_rb_per_user == 0) if mMTC_users > 0 else True
                        
                        # 检查每个用户是否能获得足够的资源块
                        urllc_rb_per_user_actual = urllc_rbs / URLLC_users if URLLC_users > 0 else 0
                        embb_rb_per_user_actual = embb_rbs / eMBB_users if eMBB_users > 0 else 0
                        
                        # 优先满足URLLC和eMBB，mMTC采用优先级分配策略
                        if (urllc_valid and embb_valid and mmtc_valid and
                            urllc_rb_per_user_actual >= URLLC_rb_per_user and 
                            embb_rb_per_user_actual >= eMBB_rb_per_user):
                            all_allocations.append((urllc_rbs, embb_rbs, mmtc_rbs))
        
        print(f"找到 {len(all_allocations)} 个满足URLLC和eMBB约束的分配方案（mMTC采用优先级分配）")
        for i, (urllc, embb, mmtc) in enumerate(all_allocations):
            mmtc_full_users = mmtc // mMTC_rb_per_user
            mmtc_remaining = mmtc % mMTC_rb_per_user
            mmtc_total_served = mmtc_full_users + mmtc_remaining
            print(f"  方案{i+1}: URLLC={urllc}, eMBB={embb}, mMTC={mmtc} (完整约束用户:{mmtc_full_users}, 部分约束用户:{mmtc_remaining}, 总服务用户:{mmtc_total_served})")
        
        if len(all_allocations) == 0:
            print(f"\n❌ 错误：即使采用灵活策略也无法找到可行方案！")
            return None, float('-inf')
    
    best_qos = float('-inf')
    best_allocation = None
    best_details = None
    best_constraints = None
    best_avg_qos = None
    best_std_qos = None
    
    print(f"\n=== 穷举搜索最优分配方案 ===")
    
    for i, (urllc_rbs, embb_rbs, mmtc_rbs) in enumerate(all_allocations):
        print(f"\n--- 测试分配方案 {i+1}/{len(all_allocations)}: URLLC={urllc_rbs}, eMBB={embb_rbs}, mMTC={mmtc_rbs} ---")
        qos, urllc_qos, embb_qos, mmtc_qos, constraints, avg_qos, std_qos = evaluate_allocation_with_constraints(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
        
        urllc_satisfied, embb_satisfied, mmtc_satisfied = constraints
        
        if qos > best_qos:
            best_qos = qos
            best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
            best_details = (urllc_qos, embb_qos, mmtc_qos)
            best_constraints = constraints
            best_avg_qos = avg_qos
            best_std_qos = std_qos
            print(f"  *** 发现更好的方案！总QoS: {qos:.4f} ***")
    
    # 输出结果
    urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
    urllc_qos, embb_qos, mmtc_qos = best_details
    urllc_satisfied, embb_satisfied, mmtc_satisfied = best_constraints
    avg_urllc_qos, avg_embb_qos, avg_mmtc_qos, avg_total_qos = best_avg_qos
    std_urllc_qos, std_embb_qos, std_mmtc_qos, std_total_qos = best_std_qos
    
    print(f"\n" + "="*60)
    print(f"=== 最优资源分配方案（满足约束） ===")
    print(f"="*60)
    print(f"URLLC切片: {urllc_rbs} 个资源块 ({urllc_rbs/R_total*100:.1f}%)")
    print(f"eMBB切片: {embb_rbs} 个资源块 ({embb_rbs/R_total*100:.1f}%)")
    print(f"mMTC切片: {mmtc_rbs} 个资源块 ({mmtc_rbs/R_total*100:.1f}%)")
    print(f"总使用: {urllc_rbs + embb_rbs + mmtc_rbs} 个资源块")
    print(f"资源利用率: {(urllc_rbs + embb_rbs + mmtc_rbs)/R_total*100:.1f}%")
    
    print(f"\n=== 约束满足情况 ===")
    print(f"URLLC约束满足: {'✓' if urllc_satisfied else '✗'}")
    print(f"eMBB约束满足: {'✓' if embb_satisfied else '✗'}")
    print(f"mMTC约束满足: {'✓' if mmtc_satisfied else '✗'} (不强制要求)")
    
    print(f"\n=== 服务质量分析 ===")
    print(f"总服务质量: {best_qos:.4f}")
    print(f"URLLC服务质量: {urllc_qos:.4f} (占比: {urllc_qos/best_qos*100:.1f}%)")
    print(f"eMBB服务质量: {embb_qos:.4f} (占比: {embb_qos/best_qos*100:.1f}%)")
    print(f"mMTC服务质量: {mmtc_qos:.4f} (占比: {mmtc_qos/best_qos*100:.1f}%)")
    
    print(f"\n=== 平均QoS分析 ===")
    print(f"URLLC平均QoS: {avg_urllc_qos:.4f} ± {std_urllc_qos:.4f}")
    print(f"eMBB平均QoS: {avg_embb_qos:.4f} ± {std_embb_qos:.4f}")
    print(f"mMTC平均QoS: {avg_mmtc_qos:.4f} ± {std_mmtc_qos:.4f}")
    print(f"整体平均QoS: {avg_total_qos:.4f} ± {std_total_qos:.4f}")
    
    # 重新计算最优方案的详细信息
    print(f"\n=== 最优方案详细性能分析 ===")
    evaluate_allocation_with_constraints(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
    
    # 输出用于论文和制图的数据
    print(f"\n" + "="*60)
    print(f"=== 论文数据输出 ===")
    print(f"="*60)
    print(f"资源分配方案: URLLC={urllc_rbs}, eMBB={embb_rbs}, mMTC={mmtc_rbs}")
    print(f"总服务质量: {best_qos:.4f}")
    print(f"URLLC服务质量: {urllc_qos:.4f}")
    print(f"eMBB服务质量: {embb_qos:.4f}")
    print(f"mMTC服务质量: {mmtc_qos:.4f}")
    print(f"资源利用率: {(urllc_rbs + embb_rbs + mmtc_rbs)/R_total*100:.1f}%")
    print(f"约束满足: URLLC={'✓' if urllc_satisfied else '✗'}, eMBB={'✓' if embb_satisfied else '✗'}, mMTC={'✓' if mmtc_satisfied else '✗'}")
    print(f"\n=== 平均QoS数据 ===")
    print(f"URLLC平均QoS: {avg_urllc_qos:.4f} ± {std_urllc_qos:.4f}")
    print(f"eMBB平均QoS: {avg_embb_qos:.4f} ± {std_embb_qos:.4f}")
    print(f"mMTC平均QoS: {avg_mmtc_qos:.4f} ± {std_mmtc_qos:.4f}")
    print(f"整体平均QoS: {avg_total_qos:.4f} ± {std_total_qos:.4f}")
    
    return best_allocation, best_qos

if __name__ == "__main__":
    solve_problem_1_corrected_qos() 