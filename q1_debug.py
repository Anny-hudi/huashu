import pandas as pd
import math

def solve_problem_1_debug():
    """解决第一题：带详细调试信息的版本"""
    
    # 系统参数
    R_total = 50
    power = 30
    bandwidth_per_rb = 360e3
    thermal_noise = -174
    NF = 7
    
    # SLA参数
    URLLC_SLA_delay = 5
    eMBB_SLA_delay = 100
    mMTC_SLA_delay = 500
    eMBB_SLA_rate = 50
    
    # 惩罚系数
    M_URLLC = 5
    M_eMBB = 3
    M_mMTC = 1
    alpha = 0.95
    
    def calculate_sinr(power_dbm, channel_gain_db, num_rbs):
        power_mw = 10**((power_dbm - 30) / 10)
        channel_gain_linear = 10**(channel_gain_db / 10)
        received_power = power_mw * channel_gain_linear
        noise_power = 10**((thermal_noise + 10*math.log10(num_rbs * bandwidth_per_rb) + NF) / 10)
        sinr = received_power / noise_power
        return sinr
    
    def calculate_rate(sinr, num_rbs):
        rate = num_rbs * bandwidth_per_rb * math.log2(1 + sinr)
        return rate / 1e6
    
    def calculate_urllc_qos(rate, delay):
        if delay <= URLLC_SLA_delay:
            return alpha ** delay
        else:
            return -M_URLLC
    
    def calculate_embb_qos(rate, delay):
        if delay <= eMBB_SLA_delay:
            if rate >= eMBB_SLA_rate:
                return 1.0
            else:
                return max(0.0, rate / eMBB_SLA_rate)
        else:
            return -M_eMBB
    
    def calculate_mmtc_qos(connection_ratio, delay):
        if delay <= mMTC_SLA_delay:
            return connection_ratio
        else:
            return -M_mMTC
    
    def evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data):
        total_qos = 0.0
        debug_info = []
        
        # URLLC用户评估
        if urllc_rbs > 0:
            for i in range(2):
                user_key = f'U{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = calculate_sinr(power, channel_gain, urllc_rbs)
                    rate = calculate_rate(sinr, urllc_rbs)
                    data_size = 0.011
                    delay = data_size / rate * 1000
                    qos = calculate_urllc_qos(rate, delay)
                    total_qos += qos
                    
                    debug_info.append({
                        'user': user_key, 'slice': 'URLLC', 'rbs': urllc_rbs,
                        'channel_gain': channel_gain, 'sinr': sinr, 'rate': rate,
                        'delay': delay, 'qos': qos
                    })
        
        # eMBB用户评估
        if embb_rbs > 0:
            for i in range(4):
                user_key = f'e{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = calculate_sinr(power, channel_gain, embb_rbs)
                    rate = calculate_rate(sinr, embb_rbs)
                    data_size = 0.11
                    delay = data_size / rate * 1000
                    qos = calculate_embb_qos(rate, delay)
                    total_qos += qos
                    
                    debug_info.append({
                        'user': user_key, 'slice': 'eMBB', 'rbs': embb_rbs,
                        'channel_gain': channel_gain, 'sinr': sinr, 'rate': rate,
                        'delay': delay, 'qos': qos
                    })
        
        # mMTC用户评估
        if mmtc_rbs > 0:
            connected_users = 0
            total_users = 0
            mmtc_details = []
            
            for i in range(10):
                user_key = f'm{i+1}'
                if user_key in user_data:
                    total_users += 1
                    channel_gain = user_data[user_key]
                    sinr = calculate_sinr(power, channel_gain, mmtc_rbs)
                    rate = calculate_rate(sinr, mmtc_rbs)
                    connected = rate >= 1
                    if connected:
                        connected_users += 1
                    mmtc_details.append({'user': user_key, 'rate': rate, 'connected': connected})
            
            connection_ratio = connected_users / total_users if total_users > 0 else 0
            data_size = 0.013
            avg_rate = mmtc_rbs * bandwidth_per_rb * math.log2(1 + 1) / 1e6
            delay = data_size / avg_rate * 1000
            qos = calculate_mmtc_qos(connection_ratio, delay)
            total_qos += qos
            
            debug_info.append({
                'slice': 'mMTC', 'rbs': mmtc_rbs, 'connection_ratio': connection_ratio,
                'delay': delay, 'qos': qos, 'details': mmtc_details
            })
        
        return total_qos, debug_info
    
    # 加载数据
    data = pd.read_excel('data1.xlsx')
    user_data = data.iloc[0]
    
    print("=== 第一题：详细调试版本 ===")
    print(f"系统参数: R_total={R_total}, power={power}dBm, bandwidth={bandwidth_per_rb/1000:.1f}kHz")
    print(f"SLA参数: URLLC延迟={URLLC_SLA_delay}ms, eMBB延迟={eMBB_SLA_delay}ms, eMBB速率={eMBB_SLA_rate}Mbps")
    print(f"惩罚系数: URLLC={M_URLLC}, eMBB={M_eMBB}, mMTC={M_mMTC}, alpha={alpha}")
    
    print(f"\n用户信道增益:")
    for key, value in user_data.items():
        if key != 'Time':
            print(f"  {key}: {value:.2f} dB")
    
    # 搜索最优方案
    print(f"\n=== 搜索最优分配方案 ===")
    
    best_qos = float('-inf')
    best_allocation = None
    best_debug_info = None
    search_count = 0
    
    urllc_possible = [0, 10, 20, 30, 40, 50]
    embb_possible = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    mmtc_possible = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    print(f"搜索空间: URLLC{len(urllc_possible)}种, eMBB{len(embb_possible)}种, mMTC{len(mmtc_possible)}种")
    
    for urllc_rbs in urllc_possible:
        for embb_rbs in embb_possible:
            for mmtc_rbs in mmtc_possible:
                if urllc_rbs + embb_rbs + mmtc_rbs == R_total:
                    search_count += 1
                    qos, debug_info = evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
                    
                    if qos > best_qos:
                        best_qos = qos
                        best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
                        best_debug_info = debug_info
                        print(f"发现更优方案 #{search_count}: URLLC({urllc_rbs}) + eMBB({embb_rbs}) + mMTC({mmtc_rbs}) = {urllc_rbs+embb_rbs+mmtc_rbs}RB, QoS={qos:.4f}")
    
    print(f"搜索完成，共检查 {search_count} 个方案")
    
    # 输出最终结果
    urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
    
    print(f"\n=== 最优资源分配方案 ===")
    print(f"URLLC切片: {urllc_rbs} RB ({urllc_rbs/R_total*100:.1f}%)")
    print(f"eMBB切片: {embb_rbs} RB ({embb_rbs/R_total*100:.1f}%)")
    print(f"mMTC切片: {mmtc_rbs} RB ({mmtc_rbs/R_total*100:.1f}%)")
    print(f"总使用: {urllc_rbs + embb_rbs + mmtc_rbs} RB (100%)")
    print(f"总服务质量: {best_qos:.4f}")
    
    # 详细分析
    print(f"\n=== 详细性能分析 ===")
    
    # URLLC分析
    urllc_users = [d for d in best_debug_info if d['slice'] == 'URLLC']
    if urllc_users:
        print(f"URLLC切片分析 ({urllc_rbs} RB):")
        for user in urllc_users:
            print(f"  {user['user']}: 信道增益={user['channel_gain']:.2f}dB, SINR={user['sinr']:.2f}, 速率={user['rate']:.2f}Mbps, 延迟={user['delay']:.4f}ms, QoS={user['qos']:.4f}")
    
    # eMBB分析
    embb_users = [d for d in best_debug_info if d['slice'] == 'eMBB']
    if embb_users:
        print(f"\neMBB切片分析 ({embb_rbs} RB):")
        for user in embb_users:
            print(f"  {user['user']}: 信道增益={user['channel_gain']:.2f}dB, SINR={user['sinr']:.2f}, 速率={user['rate']:.2f}Mbps, 延迟={user['delay']:.4f}ms, QoS={user['qos']:.4f}")
    
    # mMTC分析
    mmtc_info = [d for d in best_debug_info if d['slice'] == 'mMTC']
    if mmtc_info:
        mmtc_data = mmtc_info[0]
        print(f"\nmMTC切片分析 ({mmtc_rbs} RB):")
        print(f"  连接率: {mmtc_data['connection_ratio']:.2f}, 延迟: {mmtc_data['delay']:.4f}ms, QoS: {mmtc_data['qos']:.4f}")
        print(f"  用户详情:")
        for user_detail in mmtc_data['details']:
            status = "连接" if user_detail['connected'] else "未连接"
            print(f"    {user_detail['user']}: {user_detail['rate']:.2f} Mbps ({status})")
    
    # 性能统计
    print(f"\n=== 性能统计 ===")
    total_users = len(urllc_users) + len(embb_users) + len(mmtc_info[0]['details']) if mmtc_info else 0
    avg_qos = best_qos / total_users if total_users > 0 else 0
    
    print(f"总用户数: {total_users}")
    print(f"平均QoS: {avg_qos:.4f}")
    print(f"URLLC用户数: {len(urllc_users)}")
    print(f"eMBB用户数: {len(embb_users)}")
    print(f"mMTC用户数: {len(mmtc_info[0]['details']) if mmtc_info else 0}")
    
    return best_allocation, best_qos

if __name__ == "__main__":
    solve_problem_1_debug() 