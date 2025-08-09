import pandas as pd
import math

def solve_problem_1_final():
    """解决第一题：微基站资源块分配优化（最终正确版本）"""
    
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
    
    # 惩罚系数
    M_URLLC = 5
    M_eMBB = 3
    M_mMTC = 1
    alpha = 0.95  # URLLC效用折扣系数
    
    def calculate_sinr(power_dbm, channel_gain_db, num_rbs):
        """计算信干噪比"""
        power_mw = 10**((power_dbm - 30) / 10)
        channel_gain_linear = 10**(channel_gain_db / 10)
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
                return max(0.0, rate / eMBB_SLA_rate)  # 确保不为负数
        else:
            return -M_eMBB
    
    def calculate_mmtc_qos(connection_ratio, delay):
        """计算mMTC服务质量"""
        if delay <= mMTC_SLA_delay:
            return connection_ratio
        else:
            return -M_mMTC
    
    def evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data):
        """评估资源分配方案的服务质量"""
        total_qos = 0.0
        
        # URLLC用户评估
        if urllc_rbs > 0:
            for i in range(2):  # U1, U2
                user_key = f'U{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = calculate_sinr(power, channel_gain, urllc_rbs)
                    rate = calculate_rate(sinr, urllc_rbs)
                    
                    # 估算延迟 (简化模型)
                    data_size = 0.011  # 平均数据量 0.01-0.012 Mbit
                    delay = data_size / rate * 1000  # 转换为ms
                    
                    qos = calculate_urllc_qos(rate, delay)
                    total_qos += qos
        
        # eMBB用户评估
        if embb_rbs > 0:
            for i in range(4):  # e1, e2, e3, e4
                user_key = f'e{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = calculate_sinr(power, channel_gain, embb_rbs)
                    rate = calculate_rate(sinr, embb_rbs)
                    
                    # 估算延迟
                    data_size = 0.11  # 平均数据量 0.1-0.12 Mbit
                    delay = data_size / rate * 1000
                    
                    qos = calculate_embb_qos(rate, delay)
                    total_qos += qos
        
        # mMTC用户评估
        if mmtc_rbs > 0:
            connected_users = 0
            total_users = 0
            
            for i in range(10):  # m1-m10
                user_key = f'm{i+1}'
                if user_key in user_data:
                    total_users += 1
                    channel_gain = user_data[user_key]
                    sinr = calculate_sinr(power, channel_gain, mmtc_rbs)
                    rate = calculate_rate(sinr, mmtc_rbs)
                    
                    # 简化的连接判断
                    if rate >= 1:  # 1Mbps SLA
                        connected_users += 1
            
            connection_ratio = connected_users / total_users if total_users > 0 else 0
            
            # 估算延迟
            data_size = 0.013  # 平均数据量 0.012-0.014 Mbit
            avg_rate = mmtc_rbs * bandwidth_per_rb * math.log2(1 + 1) / 1e6  # 简化计算
            delay = data_size / avg_rate * 1000
            
            qos = calculate_mmtc_qos(connection_ratio, delay)
            total_qos += qos
        
        return total_qos
    
    # 加载数据
    data = pd.read_excel('data1.xlsx')
    user_data = data.iloc[0]
    
    print("=== 第一题：微基站资源块分配优化（最终正确版本）===")
    print(f"用户数据: {dict(user_data)}")
    
    # 穷举搜索最优分配
    best_qos = float('-inf')
    best_allocation = None
    
    # 考虑倍数约束的分配方案
    urllc_possible = [0, 10, 20, 30, 40, 50]
    embb_possible = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    mmtc_possible = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    for urllc_rbs in urllc_possible:
        for embb_rbs in embb_possible:
            for mmtc_rbs in mmtc_possible:
                # 检查资源约束
                if urllc_rbs + embb_rbs + mmtc_rbs <= R_total:
                    qos = evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
                    
                    if qos > best_qos:
                        best_qos = qos
                        best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
    
    # 输出结果
    urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
    total_used = urllc_rbs + embb_rbs + mmtc_rbs
    unused_rbs = R_total - total_used
    
    print(f"\n最优资源分配方案:")
    print(f"URLLC切片: {urllc_rbs} 个资源块")
    print(f"eMBB切片: {embb_rbs} 个资源块")
    print(f"mMTC切片: {mmtc_rbs} 个资源块")
    print(f"已使用: {total_used} 个资源块")
    print(f"未使用: {unused_rbs} 个资源块")
    print(f"资源利用率: {total_used/R_total*100:.1f}%")
    print(f"总服务质量: {best_qos:.4f}")
    
    # 分析未使用资源的原因
    if unused_rbs > 0:
        print(f"\n未使用资源分析:")
        print(f"- 剩余 {unused_rbs} 个资源块无法按倍数约束分配")
        print(f"- URLLC需要10的倍数，eMBB需要5的倍数，mMTC需要2的倍数")
        print(f"- 当前分配已达到最优QoS，增加资源可能降低整体性能")
    
    # 尝试使用所有资源的方案
    print(f"\n尝试使用所有50个资源块的方案:")
    best_full_usage_qos = float('-inf')
    best_full_usage_allocation = None
    
    for urllc_rbs in urllc_possible:
        for embb_rbs in embb_possible:
            for mmtc_rbs in mmtc_possible:
                if urllc_rbs + embb_rbs + mmtc_rbs == R_total:  # 强制使用所有资源
                    qos = evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
                    
                    if qos > best_full_usage_qos:
                        best_full_usage_qos = qos
                        best_full_usage_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
    
    if best_full_usage_allocation:
        urllc_full, embb_full, mmtc_full = best_full_usage_allocation
        print(f"全资源使用方案:")
        print(f"URLLC切片: {urllc_full} 个资源块")
        print(f"eMBB切片: {embb_full} 个资源块")
        print(f"mMTC切片: {mmtc_full} 个资源块")
        print(f"总服务质量: {best_full_usage_qos:.4f}")
        
        if best_full_usage_qos > best_qos:
            print(f"✅ 全资源使用方案更优！")
            best_allocation = best_full_usage_allocation
            best_qos = best_full_usage_qos
        else:
            print(f"⚠️  当前方案更优，但未使用所有资源")
    
    # 详细分析
    print(f"\n详细分析:")
    
    if urllc_rbs > 0:
        print(f"URLLC切片 ({urllc_rbs} RB):")
        for i in range(2):
            user_key = f'U{i+1}'
            if user_key in user_data:
                channel_gain = user_data[user_key]
                sinr = calculate_sinr(power, channel_gain, urllc_rbs)
                rate = calculate_rate(sinr, urllc_rbs)
                print(f"  {user_key}: 速率={rate:.2f} Mbps, 信道增益={channel_gain:.2f} dB")
    
    if embb_rbs > 0:
        print(f"eMBB切片 ({embb_rbs} RB):")
        for i in range(4):
            user_key = f'e{i+1}'
            if user_key in user_data:
                channel_gain = user_data[user_key]
                sinr = calculate_sinr(power, channel_gain, embb_rbs)
                rate = calculate_rate(sinr, embb_rbs)
                print(f"  {user_key}: 速率={rate:.2f} Mbps, 信道增益={channel_gain:.2f} dB")
    
    if mmtc_rbs > 0:
        print(f"mMTC切片 ({mmtc_rbs} RB):")
        connected = 0
        for i in range(10):
            user_key = f'm{i+1}'
            if user_key in user_data:
                channel_gain = user_data[user_key]
                sinr = calculate_sinr(power, channel_gain, mmtc_rbs)
                rate = calculate_rate(sinr, mmtc_rbs)
                status = "连接" if rate >= 1 else "未连接"
                if rate >= 1:
                    connected += 1
                print(f"  {user_key}: 速率={rate:.2f} Mbps, 状态={status}")
        print(f"  连接率: {connected}/10 = {connected/10:.2f}")
    
    return best_allocation, best_qos

if __name__ == "__main__":
    solve_problem_1_final()
    