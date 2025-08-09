import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

def solve_problem_1_fixed_priority():
    """
    问题一解决方案 - 修正mMTC优先级计算
    严格按照题目要求：优先处理编号靠前的用户
    """
    
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
    eMBB_SLA_rate = 50     # Mbps
    alpha = 0.95  # URLLC效用折扣系数
    
    # 惩罚系数
    M_URLLC = 5
    M_eMBB = 3
    M_mMTC = 1
    
    # 每用户资源块需求
    URLLC_rb_per_user = 10
    eMBB_rb_per_user = 5
    mMTC_rb_per_user = 2
    
    # 用户数量
    URLLC_users = 2
    eMBB_users = 4
    mMTC_users = 10
    
    print("=== 问题一：网络切片资源分配优化 ===")
    print("修正mMTC优先级计算：按用户编号排序（符合题目要求）")
    print(f"总资源块: {R_total}")
    print(f"发射功率: {power} dBm")
    print(f"用户数量: URLLC={URLLC_users}, eMBB={eMBB_users}, mMTC={mMTC_users}")
    
    # 计算最小需求
    min_URLLC_rbs = URLLC_users * URLLC_rb_per_user  # 20 RB
    min_eMBB_rbs = eMBB_users * eMBB_rb_per_user      # 20 RB
    min_mMTC_rbs = mMTC_users * mMTC_rb_per_user      # 20 RB
    
    print(f"\n=== 资源需求分析 ===")
    print(f"URLLC最小需求: {min_URLLC_rbs} RB ({URLLC_users}用户 × {URLLC_rb_per_user}RB/用户)")
    print(f"eMBB最小需求: {min_eMBB_rbs} RB ({eMBB_users}用户 × {eMBB_rb_per_user}RB/用户)")
    print(f"mMTC最小需求: {min_mMTC_rbs} RB ({mMTC_users}用户 × {mMTC_rb_per_user}RB/用户)")
    print(f"总最小需求: {min_URLLC_rbs + min_eMBB_rbs + min_mMTC_rbs} RB > 总资源: {R_total} RB")
    
    print(f"\n=== 采用修正的分配策略 ===")
    print(f"根据切片间优先级分析，采用以下策略：")
    print(f"1. 切片间优先级：eMBB > URLLC > mMTC（按QoS贡献排序）")
    print(f"2. 优先满足eMBB和URLLC的约束")
    print(f"3. mMTC根据剩余资源进行分配")
    print(f"4. 切片内部优先级：按用户编号排序")
    print(f"   - URLLC: U1 > U2")
    print(f"   - eMBB: e1 > e2 > e3 > e4")
    print(f"   - mMTC: m1 > m2 > ... > m10")
    
    def calculate_mmtc_priority_allocation_fixed(mmtc_rbs, user_data):
        """
        计算mMTC用户的优先级分配方案 - 修正版本
        严格按照题目要求：优先处理编号靠前的用户
        """
        if mmtc_rbs == 0:
            return []
        
        # 计算可以满足完整约束的用户数量
        full_constraint_users = mmtc_rbs // mMTC_rb_per_user  # 可以分配2RB的用户数
        remaining_rbs = mmtc_rbs % mMTC_rb_per_user  # 剩余资源块
        
        # 按用户编号排序（符合题目要求：优先处理编号靠前的用户）
        user_list = []
        for i in range(mMTC_users):
            user_key = f'm{i+1}'
            user_list.append({
                'user_id': i,
                'user_key': user_key,
                'user_number': i + 1  # 用户编号
            })
        
        # 按用户编号排序（编号靠前的优先）
        user_list.sort(key=lambda x: x['user_number'])
        
        # 分配资源块
        allocation = []
        for i, user_info in enumerate(user_list):
            if i < full_constraint_users:
                # 满足完整约束的用户（分配2RB）
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': mMTC_rb_per_user,  # 2RB
                    'priority_rank': i + 1,
                    'constraint_satisfied': True,
                    'allocation_type': 'full_constraint'
                })
            elif i < full_constraint_users + remaining_rbs:
                # 分配1RB的用户
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': 1,  # 1RB
                    'priority_rank': i + 1,
                    'constraint_satisfied': False,
                    'allocation_type': 'partial_constraint'
                })
            else:
                # 没有分配到资源的用户
                allocation.append({
                    'user_id': user_info['user_id'],
                    'user_key': user_info['user_key'],
                    'rbs': 0,  # 0RB
                    'priority_rank': i + 1,
                    'constraint_satisfied': False,
                    'allocation_type': 'no_allocation'
                })
        
        return allocation
    
    def calculate_sinr(power_dbm, large_scale_db, small_scale, num_rbs):
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
    
    def evaluate_allocation_with_fixed_priority(urllc_rbs, embb_rbs, mmtc_rbs, user_data):
        """评估分配方案（使用修正的mMTC优先级）"""
        print(f"\n--- 测试分配方案: URLLC={urllc_rbs}, eMBB={embb_rbs}, mMTC={mmtc_rbs} ---")
        
        # 检查资源约束
        total_rbs = urllc_rbs + embb_rbs + mmtc_rbs
        if total_rbs != R_total:
            print(f"❌ 资源约束不满足: {total_rbs} != {R_total}")
            return 0, 0, 0, 0, (False, False, False)
        
        # 检查倍数约束
        urllc_valid = urllc_rbs % URLLC_rb_per_user == 0
        embb_valid = embb_rbs % eMBB_rb_per_user == 0
        mmtc_valid = mmtc_rbs % mMTC_rb_per_user == 0
        
        print(f"倍数约束检查: URLLC={urllc_valid}, eMBB={embb_valid}, mMTC={mmtc_valid}")
        
        # 计算可服务用户数
        urllc_served = urllc_rbs // URLLC_rb_per_user
        embb_served = embb_rbs // eMBB_rb_per_user
        
        # 对于mMTC，使用修正的优先级分配
        mmtc_allocation = calculate_mmtc_priority_allocation_fixed(mmtc_rbs, user_data)
        mmtc_served = sum(1 for alloc in mmtc_allocation if alloc['rbs'] > 0)
        mmtc_full_constraint = sum(1 for alloc in mmtc_allocation if alloc['constraint_satisfied'])
        
        print(f"可服务用户数: URLLC={urllc_served}/{URLLC_users}, eMBB={embb_served}/{eMBB_users}, mMTC={mmtc_served}/{mMTC_users}")
        print(f"mMTC完整约束用户: {mmtc_full_constraint}个")
        
        # 计算各切片QoS
        urllc_qos_sum = 0
        embb_qos_sum = 0
        mmtc_qos = 0
        
        # URLLC用户评估
        print(f"\n=== URLLC用户评估 ===")
        for i in range(URLLC_users):
            user_key = f'U{i+1}'
            if i < urllc_served:
                # 分配了资源的用户
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                # 计算SINR和速率
                sinr = calculate_sinr(power, large_scale, small_scale, URLLC_rb_per_user)
                rate = calculate_rate(sinr, URLLC_rb_per_user)
                
                # 计算延迟（仅传输延迟）
                delay = task_size / rate * 1000  # 转换为ms
                
                # 计算QoS
                qos = calculate_urllc_qos(rate, delay)
                urllc_qos_sum += qos
                
                print(f"  {user_key}: 速率={rate:.2f}Mbps, 延迟={delay:.4f}ms, QoS={qos:.4f}")
            else:
                print(f"  {user_key}: 未分配资源")
        
        # eMBB用户评估
        print(f"\n=== eMBB用户评估 ===")
        for i in range(eMBB_users):
            user_key = f'e{i+1}'
            if i < embb_served:
                # 分配了资源的用户
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                # 计算SINR和速率
                sinr = calculate_sinr(power, large_scale, small_scale, eMBB_rb_per_user)
                rate = calculate_rate(sinr, eMBB_rb_per_user)
                
                # 计算延迟（仅传输延迟）
                delay = task_size / rate * 1000  # 转换为ms
                
                # 计算QoS
                qos = calculate_embb_qos(rate, delay)
                embb_qos_sum += qos
                
                print(f"  {user_key}: 速率={rate:.2f}Mbps, 延迟={delay:.4f}ms, QoS={qos:.4f}")
            else:
                print(f"  {user_key}: 未分配资源")
        
        # mMTC用户评估（使用修正的优先级分配）
        print(f"\n=== mMTC用户评估（按用户编号优先级） ===")
        print(f"{'用户':<6} {'优先级':<6} {'分配RB':<8} {'速率(Mbps)':<12} {'延迟(ms)':<12} {'服务质量':<10} {'分配类型':<15}")
        
        total_mmtc_tasks = sum(1 for i in range(mMTC_users) if user_data['task_flow'][f'm{i+1}'] > 0)
        success_count = 0
        has_delay_exceed = False
        
        for alloc in mmtc_allocation:
            user_key = alloc['user_key']
            allocated_rbs = alloc['rbs']
            priority_rank = alloc['priority_rank']
            
            if allocated_rbs > 0:
                # 分配了资源的用户
                large_scale = user_data['large_scale'][user_key]
                small_scale = user_data['small_scale'][user_key]
                task_size = user_data['task_flow'][user_key]
                
                # 计算SINR和速率
                sinr = calculate_sinr(power, large_scale, small_scale, allocated_rbs)
                rate = calculate_rate(sinr, allocated_rbs)
                
                # 计算延迟（仅传输延迟）
                delay = task_size / rate * 1000  # 转换为ms
                
                # 检查延迟是否超标
                if delay > mMTC_SLA_delay:
                    has_delay_exceed = True
                    mmtc_user_qos = -M_mMTC
                else:
                    if task_size > 0:  # 有任务
                        success_count += 1
                    mmtc_user_qos = 0  # 单个用户不计算QoS，只计算整体接入比例
                
                print(f"{user_key:<6} {priority_rank:<6} {allocated_rbs:<8} {rate:<12.2f} {delay:<12.4f} {mmtc_user_qos:<10.4f} {alloc['allocation_type']:<15}")
            else:
                print(f"{user_key:<6} {priority_rank:<6} {allocated_rbs:<8} {'0.00':<12} {'∞':<12} {'0.0000':<10} {alloc['allocation_type']:<15}")
        
        # 计算mMTC整体QoS
        if total_mmtc_tasks == 0:
            mmtc_qos = 0.0
        else:
            if has_delay_exceed:
                mmtc_qos = -M_mMTC  # 有延迟超标，惩罚
            else:
                mmtc_qos = success_count / total_mmtc_tasks  # 接入比例
        
        print(f"\nmMTC整体QoS: {mmtc_qos:.4f}")
        print(f"成功接入用户数: {success_count}/{total_mmtc_tasks}")
        print(f"延迟超标: {'是' if has_delay_exceed else '否'}")
        
        # 计算总QoS
        total_qos = urllc_qos_sum + embb_qos_sum + mmtc_qos
        
        # 检查约束满足情况
        urllc_satisfied = urllc_served >= URLLC_users
        embb_satisfied = embb_served >= eMBB_users
        mmtc_satisfied = mmtc_full_constraint >= mMTC_users
        
        print(f"\n=== 约束满足情况 ===")
        print(f"URLLC约束满足: {'✓' if urllc_satisfied else '✗'} ({urllc_served}/{URLLC_users})")
        print(f"eMBB约束满足: {'✓' if embb_satisfied else '✗'} ({embb_served}/{eMBB_users})")
        print(f"mMTC约束满足: {'✓' if mmtc_satisfied else '✗'} ({mmtc_full_constraint}/{mMTC_users})")
        
        print(f"\n=== QoS总结 ===")
        print(f"URLLC QoS: {urllc_qos_sum:.4f}")
        print(f"eMBB QoS: {embb_qos_sum:.4f}")
        print(f"mMTC QoS: {mmtc_qos:.4f}")
        print(f"总QoS: {total_qos:.4f}")
        
        return total_qos, urllc_qos_sum, embb_qos_sum, mmtc_qos, (urllc_satisfied, embb_satisfied, mmtc_satisfied)
    
    # 加载用户数据
    print(f"\n=== 加载用户数据 ===")
    try:
        # 加载各个数据文件
        task_flow_data = pd.read_csv('data_1/任务流.csv')
        large_scale_data = pd.read_csv('data_1/大规模衰减.csv')
        small_scale_data = pd.read_csv('data_1/小规模瑞丽衰减.csv')
        user_pos_data = pd.read_csv('data_1/用户位置.csv')
        
        user_data = {
            'large_scale': {},
            'small_scale': {},
            'task_flow': {},
            'user_pos': {}
        }
        
        # 从第一行数据中提取用户信息（Time=0）
        row = task_flow_data.iloc[0]
        
        # 用户列表
        users = ['U1', 'U2', 'e1', 'e2', 'e3', 'e4', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
        
        for user in users:
            # 任务流数据
            user_data['task_flow'][user] = task_flow_data.iloc[0][user]
            
            # 大规模衰减数据
            user_data['large_scale'][user] = large_scale_data.iloc[0][user]
            
            # 小规模瑞丽衰减数据
            user_data['small_scale'][user] = small_scale_data.iloc[0][user]
            
            # 用户位置数据
            user_data['user_pos'][f'{user}_X'] = user_pos_data.iloc[0][f'{user}_X']
            user_data['user_pos'][f'{user}_Y'] = user_pos_data.iloc[0][f'{user}_Y']
        
        print("用户数据加载成功")
        
        # 显示mMTC用户优先级排序
        print(f"\n=== mMTC用户优先级排序（按用户编号） ===")
        mmtc_priority_list = []
        for i in range(mMTC_users):
            user_key = f'm{i+1}'
            user_x = user_data['user_pos'][f'{user_key}_X']
            user_y = user_data['user_pos'][f'{user_key}_Y']
            distance = math.sqrt(user_x**2 + user_y**2)
            task_size = user_data['task_flow'][user_key]
            
            mmtc_priority_list.append({
                'user_key': user_key,
                'user_number': i + 1,
                'distance': distance,
                'task_size': task_size
            })
        
        # 按用户编号排序
        mmtc_priority_list.sort(key=lambda x: x['user_number'])
        
        print(f"{'排名':<4} {'用户':<6} {'用户编号':<8} {'距离(m)':<10} {'任务数据量(Mbit)':<15}")
        for i, info in enumerate(mmtc_priority_list):
            print(f"{i+1:<4} {info['user_key']:<6} {info['user_number']:<8} {info['distance']:<10.1f} {info['task_size']:<15.6f}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 测试分配方案
    print(f"\n=== 测试分配方案 ===")
    
    # 根据切片间优先级（eMBB > URLLC > mMTC）设计分配方案
    test_allocations = [
        (20, 20, 10),  # 优先满足eMBB和URLLC，mMTC分配剩余
        (20, 25, 5),   # 优先满足eMBB，然后URLLC，mMTC最少
        (15, 25, 10),  # 优先满足eMBB，URLLC部分满足，mMTC分配剩余
        (25, 15, 10),  # 优先满足URLLC，eMBB部分满足，mMTC分配剩余
        (30, 15, 5),   # 优先满足URLLC，eMBB部分满足，mMTC最少
    ]
    
    best_qos = float('-inf')
    best_allocation = None
    best_details = None
    best_constraints = None
    
    # 存储所有有效分配方案的结果
    all_results = []
    
    for urllc_rbs, embb_rbs, mmtc_rbs in test_allocations:
        if urllc_rbs + embb_rbs + mmtc_rbs == R_total:
            qos, urllc_qos, embb_qos, mmtc_qos, constraints = evaluate_allocation_with_fixed_priority(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
            
            # 记录所有结果
            all_results.append({
                'allocation': (urllc_rbs, embb_rbs, mmtc_rbs),
                'qos': qos,
                'urllc_qos': urllc_qos,
                'embb_qos': embb_qos,
                'mmtc_qos': mmtc_qos,
                'constraints': constraints
            })
            
            if qos > best_qos:
                best_qos = qos
                best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
                best_details = (urllc_qos, embb_qos, mmtc_qos)
                best_constraints = constraints
    
    # 按QoS大小排序所有结果
    all_results.sort(key=lambda x: x['qos'], reverse=True)
    
    print(f"\n=== 按QoS排序的所有分配方案 ===")
    print(f"{'排名':<4} {'URLLC':<8} {'eMBB':<8} {'mMTC':<8} {'总QoS':<10} {'URLLC_QoS':<12} {'eMBB_QoS':<12} {'mMTC_QoS':<12}")
    for i, result in enumerate(all_results):
        urllc_rbs, embb_rbs, mmtc_rbs = result['allocation']
        print(f"{i+1:<4} {urllc_rbs:<8} {embb_rbs:<8} {mmtc_rbs:<8} {result['qos']:<10.4f} {result['urllc_qos']:<12.4f} {result['embb_qos']:<12.4f} {result['mmtc_qos']:<12.4f}")
    
    # 分析切片间优先级
    print(f"\n=== 切片间优先级分析 ===")
    
    # 计算各切片的平均QoS贡献
    avg_urllc_qos = sum(r['urllc_qos'] for r in all_results) / len(all_results) if all_results else 0
    avg_embb_qos = sum(r['embb_qos'] for r in all_results) / len(all_results) if all_results else 0
    avg_mmtc_qos = sum(r['mmtc_qos'] for r in all_results) / len(all_results) if all_results else 0
    
    print(f"平均URLLC QoS贡献: {avg_urllc_qos:.4f}")
    print(f"平均eMBB QoS贡献: {avg_embb_qos:.4f}")
    print(f"平均mMTC QoS贡献: {avg_mmtc_qos:.4f}")
    
    # 按平均QoS贡献排序切片优先级
    slice_priorities = [
        ('URLLC', avg_urllc_qos),
        ('eMBB', avg_embb_qos),
        ('mMTC', avg_mmtc_qos)
    ]
    slice_priorities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n切片间优先级（按QoS贡献排序）:")
    for i, (slice_name, qos_contribution) in enumerate(slice_priorities):
        print(f"{i+1}. {slice_name}: {qos_contribution:.4f}")
    
    # 输出最优结果
    if best_allocation:
        urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
        urllc_qos, embb_qos, mmtc_qos = best_details
        urllc_satisfied, embb_satisfied, mmtc_satisfied = best_constraints
        
        print(f"\n=== 最优分配方案 ===")
        print(f"URLLC: {urllc_rbs} RB ({urllc_rbs/R_total*100:.1f}%)")
        print(f"eMBB: {embb_rbs} RB ({embb_rbs/R_total*100:.1f}%)")
        print(f"mMTC: {mmtc_rbs} RB ({mmtc_rbs/R_total*100:.1f}%)")
        print(f"总使用: {urllc_rbs + embb_rbs + mmtc_rbs} 个资源块")
        print(f"资源利用率: {(urllc_rbs + embb_rbs + mmtc_rbs)/R_total*100:.1f}%")
        
        print(f"\n=== 最优QoS结果 ===")
        print(f"URLLC QoS: {urllc_qos:.4f}")
        print(f"eMBB QoS: {embb_qos:.4f}")
        print(f"mMTC QoS: {mmtc_qos:.4f}")
        print(f"总QoS: {best_qos:.4f}")
        
        print(f"\n=== 约束满足情况 ===")
        print(f"URLLC约束满足: {'✓' if urllc_satisfied else '✗'}")
        print(f"eMBB约束满足: {'✓' if embb_satisfied else '✗'}")
        print(f"mMTC约束满足: {'✓' if mmtc_satisfied else '✗'}")
        
        print(f"\n=== 优先级策略总结 ===")
        print(f"✅ 切片内部优先级：")
        print(f"   - URLLC: 按用户编号排序（U1 > U2）")
        print(f"   - eMBB: 按用户编号排序（e1 > e2 > e3 > e4）")
        print(f"   - mMTC: 按用户编号排序（m1 > m2 > ... > m10）")
        print(f"✅ 切片间优先级：按QoS贡献排序")
        for i, (slice_name, qos_contribution) in enumerate(slice_priorities):
            print(f"   {i+1}. {slice_name}: {qos_contribution:.4f}")
        
    else:
        print("❌ 未找到有效的分配方案")

if __name__ == "__main__":
    solve_problem_1_fixed_priority() 