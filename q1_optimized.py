import pandas as pd
import numpy as np
import math
from collections import defaultdict

# 1. 参数设置
SLA = {
    'URLLC': {
        'L_SLA': 5,       # 最大延迟(ms)
        'alpha': 0.95,    # 效用折扣系数
        'M': 5,           # 惩罚值
        'rb_per_user': 10 # 每用户资源块
    },
    'eMBB': {
        'L_SLA': 100,     # 最大延迟(ms)
        'r_SLA': 50,      # 速率SLA(Mbps)
        'M': 3,           # 惩罚值
        'rb_per_user': 5  # 每用户资源块
    },
    'mMTC': {
        'L_SLA': 500,     # 最大延迟(ms)
        'M': 1,           # 惩罚值
        'rb_per_user': 2  # 每用户资源块
    }
}
TOTAL_RB = 50
RB_BANDWIDTH = 360e3  # 360kHz/资源块
NF = 7  # 噪声系数
P_TX_DBM = 20  # 发射功率(dBm)

# 2. 模拟用户数据（实际使用时替换为Excel数据）
def generate_user_data():
    """生成模拟用户数据，实际应用中应从Excel读取"""
    user_data = [
        # URLLC用户
        {'用户编号': 'U1', 'type': 'URLLC', '大尺度衰减_dB': 80, '小尺度衰减': 0.9, '任务量_Mbit': 0.01, 'c_i': 0},
        {'用户编号': 'U2', 'type': 'URLLC', '大尺度衰减_dB': 82, '小尺度衰减': 0.85, '任务量_Mbit': 0.008, 'c_i': 0},
        
        # eMBB用户
        {'用户编号': 'e1', 'type': 'eMBB', '大尺度衰减_dB': 75, '小尺度衰减': 0.92, '任务量_Mbit': 0.11, 'c_i': 0},
        {'用户编号': 'e2', 'type': 'eMBB', '大尺度衰减_dB': 77, '小尺度衰减': 0.88, '任务量_Mbit': 0.11, 'c_i': 0},
        {'用户编号': 'e3', 'type': 'eMBB', '大尺度衰减_dB': 85, '小尺度衰减': 0.75, '任务量_Mbit': 0.10, 'c_i': 0},
        {'用户编号': 'e4', 'type': 'eMBB', '大尺度衰减_dB': 72, '小尺度衰减': 0.95, '任务量_Mbit': 0.21, 'c_i': 0},
        
        # mMTC用户
        {'用户编号': 'm1', 'type': 'mMTC', '大尺度衰减_dB': 90, '小尺度衰减': 0.6, '任务量_Mbit': 0.012, 'c_i': 1},
        {'用户编号': 'm2', 'type': 'mMTC', '大尺度衰减_dB': 88, '小尺度衰减': 0.65, '任务量_Mbit': 0.011, 'c_i': 1},
        {'用户编号': 'm3', 'type': 'mMTC', '大尺度衰减_dB': 92, '小尺度衰减': 0.58, '任务量_Mbit': 0.013, 'c_i': 1},
        {'用户编号': 'm4', 'type': 'mMTC', '大尺度衰减_dB': 87, '小尺度衰减': 0.68, '任务量_Mbit': 0.010, 'c_i': 1},
        {'用户编号': 'm5', 'type': 'mMTC', '大尺度衰减_dB': 89, '小尺度衰减': 0.63, '任务量_Mbit': 0.012, 'c_i': 1},
        {'用户编号': 'm6', 'type': 'mMTC', '大尺度衰减_dB': 86, '小尺度衰减': 0.70, '任务量_Mbit': 0.011, 'c_i': 1},
        {'用户编号': 'm7', 'type': 'mMTC', '大尺度衰减_dB': 91, '小尺度衰减': 0.59, '任务量_Mbit': 0.013, 'c_i': 1},
        {'用户编号': 'm8', 'type': 'mMTC', '大尺度衰减_dB': 85, '小尺度衰减': 0.72, '任务量_Mbit': 0.014, 'c_i': 1},
        {'用户编号': 'm9', 'type': 'mMTC', '大尺度衰减_dB': 93, '小尺度衰减': 0.57, '任务量_Mbit': 0.015, 'c_i': 1},
        {'用户编号': 'm10', 'type': 'mMTC', '大尺度衰减_dB': 90, '小尺度衰减': 0.59, '任务量_Mbit': 0.013, 'c_i': 1},
    ]
    return pd.DataFrame(user_data)

# 3. 计算单用户性能指标
def calculate_user_performance(user, rb_allocated, prev_delay=0):
    """计算用户的性能指标：时延、速率、服务质量等"""
    utype = user['type']
    params = SLA[utype]
    
    # 资源不足情况
    if rb_allocated < params['rb_per_user']:
        return {
            '排队时延': 0,
            '传输时延': 0,
            '总时延': np.inf,
            '速率(Mbps)': 0,
            '服务质量': -params['M'],
            '是否服务': False
        }
    
    # 传输参数计算
    phi = user['大尺度衰减_dB']
    h = user['小尺度衰减']
    data_vol = user['任务量_Mbit']
    i_rb = rb_allocated
    
    # 接收功率计算
    p_rx = 10 ** ((P_TX_DBM - phi) / 10) * (h ** 2)
    
    # 噪声功率计算
    n0_db = -174 + 10 * np.log10(i_rb * RB_BANDWIDTH) + NF
    n0_mw = 10 **(n0_db / 10)
    
    # 信噪比与速率计算
    snr = p_rx / n0_mw if n0_mw != 0 else 0
    rate = i_rb * RB_BANDWIDTH * np.log2(1 + snr) / 1e6 if snr >= 0 else 0
    
    # 延迟计算
    transmission_delay = (data_vol / rate) * 1000 if rate > 0 else np.inf
    queue_delay = prev_delay  # 排队延迟基于前一个用户的总延迟
    total_delay = transmission_delay + queue_delay
    
    # 计算服务质量
    if utype == 'URLLC':
        if total_delay <= params['L_SLA']:
            qos = params['alpha']** total_delay
        else:
            qos = -params['M']
        is_served = True
        
    elif utype == 'eMBB':
        if total_delay > params['L_SLA']:
            qos = -params['M']
            is_served = False
        else:
            if rate >= params['r_SLA']:
                qos = 1.0
            else:
                qos = rate / params['r_SLA']
            is_served = True
            
    elif utype == 'mMTC':
        qos = 1.0 if (total_delay <= params['L_SLA'] and user['c_i'] == 1) else 0.0
        is_served = (total_delay <= params['L_SLA'])
    
    return {
        '排队时延': queue_delay,
        '传输时延': transmission_delay,
        '总时延': total_delay,
        '速率(Mbps)': rate,
        '服务质量': qos,
        '是否服务': is_served
    }

# 4. 计算分配方案的总服务质量
def calculate_allocation_quality(ru, re, rm, df):
    """计算给定资源分配方案的服务质量"""
    # 按切片分组用户
    urllc_users = df[df['type'] == 'URLLC'].sort_values('大尺度衰减_dB')  # 按信道质量排序
    embb_users = df[df['type'] == 'eMBB'].sort_values('大尺度衰减_dB')
    mmtc_users = df[df['type'] == 'mMTC'].sort_values('大尺度衰减_dB')
    
    # 计算各切片可服务的用户数
    urllc_count = min(ru // SLA['URLLC']['rb_per_user'], len(urllc_users))
    embb_count = min(re // SLA['eMBB']['rb_per_user'], len(embb_users))
    mmtc_count = min(rm // SLA['mMTC']['rb_per_user'], len(mmtc_users))
    
    # 计算每个用户的性能
    user_results = {}
    prev_delay = 0
    
    # URLLC用户
    for i, (_, user) in enumerate(urllc_users.iterrows()):
        rb = SLA['URLLC']['rb_per_user'] if i < urllc_count else 0
        results = calculate_user_performance(user, rb, prev_delay)
        user_results[user['用户编号']] = results
        if i < urllc_count:
            prev_delay = results['总时延']
    
    # eMBB用户
    prev_delay = 0
    for i, (_, user) in enumerate(embb_users.iterrows()):
        rb = SLA['eMBB']['rb_per_user'] if i < embb_count else 0
        results = calculate_user_performance(user, rb, prev_delay)
        user_results[user['用户编号']] = results
        if i < embb_count:
            prev_delay = results['总时延']
    
    # mMTC用户
    prev_delay = 0
    mmtc_qualified = 0
    mmtc_total_tasks = sum(mmtc_users['c_i'])
    for i, (_, user) in enumerate(mmtc_users.iterrows()):
        rb = SLA['mMTC']['rb_per_user'] if i < mmtc_count else 0
        results = calculate_user_performance(user, rb, prev_delay)
        user_results[user['用户编号']] = results
        if i < mmtc_count and user['c_i'] == 1 and results['总时延'] <= SLA['mMTC']['L_SLA']:
            mmtc_qualified += 1
        if i < mmtc_count:
            prev_delay = results['总时延']
    
    # 计算各切片总服务质量
    urllc_qos = sum(result['服务质量'] for uid, result in user_results.items() 
                   if df[df['用户编号'] == uid]['type'].values[0] == 'URLLC')
    
    embb_qos = sum(result['服务质量'] for uid, result in user_results.items() 
                  if df[df['用户编号'] == uid]['type'].values[0] == 'eMBB')
    
    # mMTC服务质量特殊计算（整体接入比例）
    if mmtc_total_tasks > 0:
        if mmtc_qualified == mmtc_total_tasks:
            mmtc_qos = mmtc_qualified / mmtc_total_tasks
        else:
            mmtc_qos = -SLA['mMTC']['M']
    else:
        mmtc_qos = 0
    
    total_qos = urllc_qos + embb_qos + mmtc_qos
    
    return {
        '总服务质量': total_qos,
        'urllc_qos': urllc_qos,
        'embb_qos': embb_qos,
        'mmtc_qos': mmtc_qos,
        'user_results': user_results,
        'urllc_count': urllc_count,
        'embb_count': embb_count,
        'mmtc_count': mmtc_count
    }

# 5. 枚举所有可能的资源分配方案
def enumerate_all_allocations(df):
    """枚举所有可能的资源分配方案并找到最优解"""
    print("开始枚举所有可能的资源分配方案...")
    
    results = []
    urllc_rb_step = SLA['URLLC']['rb_per_user']
    embb_rb_step = SLA['eMBB']['rb_per_user']
    mmtc_rb_step = SLA['mMTC']['rb_per_user']
    
    # 枚举所有可能的分配方案
    for ru in range(0, TOTAL_RB + 1, urllc_rb_step):
        for re in range(0, TOTAL_RB - ru + 1, embb_rb_step):
            rm = TOTAL_RB - ru - re
            if rm < 0:
                continue
            if rm % mmtc_rb_step != 0:
                continue
                
            # 计算该方案的服务质量
            quality_data = calculate_allocation_quality(ru, re, rm, df)
            
            results.append({
                'RU': ru,
                'Re': re,
                'Rm': rm,
                '总服务质量': quality_data['总服务质量'],
                '详细数据': quality_data
            })
    
    # 按总服务质量排序
    results.sort(key=lambda x: x['总服务质量'], reverse=True)
    return results

# 6. 格式化输出结果
def format_output(results, df):
    """按照指定格式输出结果"""
    if not results:
        print("没有找到有效的资源分配方案")
        return
    
    # 最优方案
    best = results[0]
    print("优化完成！")
    print(f"最优资源分配方案: RU={best['RU']}, Re={best['Re']}, Rm={best['Rm']}")
    print(f"最大服务质量: {best['总服务质量']:.4f}\n")
    
    # 详细分析
    details = best['详细数据']
    print("=" * 60)
    print("最优资源分配方案详细分析")
    print("=" * 60)
    print(f"URLLC切片分配: {best['RU']} RB (并发用户数: {details['urllc_count']})")
    print(f"eMBB切片分配: {best['Re']} RB (并发用户数: {details['embb_count']})")
    print(f"mMTC切片分配: {best['Rm']} RB (并发用户数: {details['mmtc_count']})")
    print(f"总服务质量: {best['总服务质量']:.4f}\n")
    
    # 用户详细结果表格
    print("各用户详细结果:")
    print("-" * 100)
    print(f"{'用户':<8}{'切片':<10}{'排队时延':<12}{'传输时延':<12}{'总时延':<12}{'速率(Mbps)':<15}{'服务质量':<12}{'是否服务'}")
    print("-" * 100)
    
    # 按用户类型和编号排序
    user_types = {uid: df[df['用户编号'] == uid]['type'].values[0] for uid in df['用户编号']}
    sorted_users = sorted(df['用户编号'], key=lambda x: (user_types[x], x))
    
    for uid in sorted_users:
        utype = user_types[uid]
        res = details['user_results'][uid]
        
        # 格式化输出
        print(f"{uid:<8}{utype:<10}"
              f"{res['排队时延']:<12.2f}{res['传输时延']:<12.2f}{res['总时延']:<12.2f}"
              f"{res['速率(Mbps)']:<15.2f}{res['服务质量']:<12.4f}"
              f"{'是' if res['是否服务'] else '否'}")
    
    print("-" * 100 + "\n")
    
    # 各切片汇总
    print("各切片汇总:")
    urllc_users_count = len(df[df['type'] == 'URLLC'])
    embb_users_count = len(df[df['type'] == 'eMBB'])
    mmtc_users_count = len(df[df['type'] == 'mMTC'])
    
    print(f"URLLC: 平均服务质量 = {details['urllc_qos']/urllc_users_count:.4f}, 总服务质量 = {details['urllc_qos']:.4f}")
    print(f"eMBB: 平均服务质量 = {details['embb_qos']/embb_users_count:.4f}, 总服务质量 = {details['embb_qos']:.4f}")
    print(f"mMTC: 平均服务质量 = {details['mmtc_qos']:.4f}, 总服务质量 = {details['mmtc_qos']:.4f}\n")
    
    # Top 10最优方案
    print("Top 10最优资源分配方案:")
    print(f"{'RU':<5}{'Re':<5}{'Rm':<5}{'Total_QoS':<10}")
    for i, res in enumerate(results[:10]):
        print(f"{res['RU']:<5}{res['Re']:<5}{res['Rm']:<5}{res['总服务质量']:.6f}")

# 7. 主函数
def main():
    # 生成用户数据（实际应用中替换为Excel读取）
    df = generate_user_data()
    
    # 枚举所有分配方案
    allocation_results = enumerate_all_allocations(df)
    
    # 格式化输出结果
    format_output(allocation_results, df)

if __name__ == "__main__":
    main()
    