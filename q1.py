import pandas as pd
import numpy as np
import math
from collections import defaultdict

# 1. 参数设置（匹配题目定义）
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


# 2. 读取用户数据（从Excel读取，替换模拟数据）
def load_user_data(file_path):
    excel_file = pd.ExcelFile(file_path)
    df_large = excel_file.parse('大规模衰减')
    df_small = excel_file.parse('小规模瑞丽衰减')
    df_location = excel_file.parse('用户位置')
    df_task = excel_file.parse('任务流')
    
    user_ids = ['U1', 'U2', 'e1', 'e2', 'e3', 'e4', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
    data = []
    f = 2000  # 载波频率(MHz)
    
    for uid in user_ids:
        utype = 'URLLC' if uid.startswith('U') else 'eMBB' if uid.startswith('e') else 'mMTC'
        x = df_location[f'{uid}_X'].values[0]
        y = df_location[f'{uid}_Y'].values[0]
        d_m = math.sqrt(x**2 + y**2)
        d_km = d_m / 1000
        
        # 计算大尺度衰减（自由空间模型）
        large_scale = 32.45 + 20 * math.log10(d_km) + 20 * math.log10(f) if d_km > 0 else 0
        small_scale = df_small[uid].values[0]
        task_vol = df_task[uid].values[0]
        c_i = 1 if utype == 'mMTC' else 0  # mMTC有任务标记
        
        data.append({
            '用户编号': uid,
            'type': utype,
            '大尺度衰减_dB': large_scale,
            '小尺度衰减': small_scale,
            '任务量_Mbit': task_vol,
            'c_i': c_i
        })
    return pd.DataFrame(data)


# 3. 计算单用户性能指标
def calculate_user_performance(user, rb_allocated, prev_delay=0):
    utype = user['type']
    params = SLA[utype]
    
    # 资源不足：接入失败（仅记录状态，不直接惩罚）
    if rb_allocated < params['rb_per_user']:
        return {
            '排队时延': 0,
            '传输时延': 0,
            '总时延': np.inf,  # 标记为未接入
            '速率(Mbps)': 0,
            '服务质量': 0,  # 暂不计算惩罚
            '是否服务': False,
            '是否延迟超标': False  # 未接入不算延迟超标
        }
    
    # 传输参数计算
    phi = user['大尺度衰减_dB']
    h = user['小尺度衰减']
    data_vol = user['任务量_Mbit']
    i_rb = rb_allocated
    
    # 接收功率与噪声
    p_rx = 10 ** ((P_TX_DBM - phi) / 10) * (h ** 2)
    n0_db = -174 + 10 * np.log10(i_rb * RB_BANDWIDTH) + NF
    n0_mw = 10 **(n0_db / 10)
    
    # 速率与延迟
    snr = p_rx / n0_mw if n0_mw != 0 else 0
    rate = i_rb * RB_BANDWIDTH * np.log2(1 + snr) / 1e6 if snr >= 0 else 0
    transmission_delay = (data_vol / rate) * 1000 if rate > 0 else np.inf
    queue_delay = prev_delay
    total_delay = transmission_delay + queue_delay
    is_delay_exceed = total_delay > params['L_SLA']
    
    # 计算服务质量
    if utype == 'URLLC':
        qos = params['alpha']** total_delay if not is_delay_exceed else -params['M']
        return {
            '排队时延': queue_delay,
            '传输时延': transmission_delay,
            '总时延': total_delay,
            '速率(Mbps)': rate,
            '服务质量': qos,
            '是否服务': True,
            '是否延迟超标': is_delay_exceed
        }
    
    elif utype == 'eMBB':
        if is_delay_exceed:
            qos = -params['M']
        else:
            qos = 1.0 if rate >= params['r_SLA'] else (rate / params['r_SLA'])
        return {
            '排队时延': queue_delay,
            '传输时延': transmission_delay,
            '总时延': total_delay,
            '速率(Mbps)': rate,
            '服务质量': qos,
            '是否服务': True,
            '是否延迟超标': is_delay_exceed
        }
    
    elif utype == 'mMTC':
        # mMTC的QoS后续整体计算，这里仅记录状态
        return {
            '排队时延': queue_delay,
            '传输时延': transmission_delay,
            '总时延': total_delay,
            '速率(Mbps)': rate,
            '服务质量': 0,  # 暂不计算
            '是否服务': True,
            '是否延迟超标': is_delay_exceed
        }


# 4. 计算分配方案的总服务质量（核心修正：mMTC惩罚逻辑）
def calculate_allocation_quality(ru, re, rm, df):
    # 按切片分组用户（按编号顺序，符合题目优先级）
    urllc_users = df[df['type'] == 'URLLC'].sort_values('用户编号')
    embb_users = df[df['type'] == 'eMBB'].sort_values('用户编号')
    mmtc_users = df[df['type'] == 'mMTC'].sort_values('用户编号')
    
    # 可服务用户数
    urllc_count = min(ru // SLA['URLLC']['rb_per_user'], len(urllc_users))
    embb_count = min(re // SLA['eMBB']['rb_per_user'], len(embb_users))
    mmtc_count = min(rm // SLA['mMTC']['rb_per_user'], len(mmtc_users))
    
    # 计算每个用户的性能
    user_results = {}
    prev_delay = 0  # 排队时延基于前一个用户的总时延
    
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
    mmtc_results = []
    for i, (_, user) in enumerate(mmtc_users.iterrows()):
        rb = SLA['mMTC']['rb_per_user'] if i < mmtc_count else 0
        results = calculate_user_performance(user, rb, prev_delay)
        user_results[user['用户编号']] = results
        mmtc_results.append(results)
        if i < mmtc_count:
            prev_delay = results['总时延']
    
    # 计算各切片QoS
    # URLLC：直接累加单用户得分
    urllc_qos = sum(result['服务质量'] for uid, result in user_results.items() 
                   if df[df['用户编号'] == uid]['type'].values[0] == 'URLLC')
    
    # eMBB：直接累加单用户得分
    embb_qos = sum(result['服务质量'] for uid, result in user_results.items() 
                  if df[df['用户编号'] == uid]['type'].values[0] == 'eMBB')
    
    # mMTC：修正惩罚逻辑（仅当有用户延迟超标时惩罚）
    total_mmtc_tasks = sum(mmtc_users['c_i'])  # 有任务的用户总数
    if total_mmtc_tasks == 0:
        mmtc_qos = 0.0
    else:
        # 成功接入且延迟达标的用户数（c_i'=1）
        success_count = sum(1 for res in mmtc_results 
                          if res['是否服务'] and not res['是否延迟超标'] and mmtc_users.iloc[mmtc_results.index(res)]['c_i'] == 1)
        # 检查是否有用户延迟超标（分配了资源但延迟> SLA）
        has_exceed = any(res['是否服务'] and res['是否延迟超标'] for res in mmtc_results)
        
        if has_exceed:
            mmtc_qos = -SLA['mMTC']['M']  # 有延迟超标，惩罚
        else:
            mmtc_qos = success_count / total_mmtc_tasks  # 接入比例
    
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


# 5. 枚举所有分配方案
def enumerate_all_allocations(df):
    print("开始枚举所有可能的资源分配方案...")
    
    results = []
    urllc_step = SLA['URLLC']['rb_per_user']
    embb_step = SLA['eMBB']['rb_per_user']
    mmtc_step = SLA['mMTC']['rb_per_user']
    
    for ru in range(0, TOTAL_RB + 1, urllc_step):
        for re in range(0, TOTAL_RB - ru + 1, embb_step):
            rm = TOTAL_RB - ru - re
            if rm < 0 or rm % mmtc_step != 0:
                continue
            
            quality_data = calculate_allocation_quality(ru, re, rm, df)
            results.append({
                'RU': ru, 'Re': re, 'Rm': rm,
                '总服务质量': quality_data['总服务质量'],
                '详细数据': quality_data
            })
    
    results.sort(key=lambda x: x['总服务质量'], reverse=True)
    return results


# 6. 格式化输出
def format_output(results, df):
    if not results:
        print("无有效方案")
        return
    
    best = results[0]
    print("优化完成！")
    print(f"最优资源分配方案: RU={best['RU']}, Re={best['Re']}, Rm={best['Rm']}")
    print(f"最大服务质量: {best['总服务质量']:.4f}\n")
    
    details = best['详细数据']
    print("=" * 60)
    print("最优资源分配方案详细分析")
    print("=" * 60)
    print(f"URLLC切片分配: {best['RU']} RB (并发用户数: {details['urllc_count']})")
    print(f"eMBB切片分配: {best['Re']} RB (并发用户数: {details['embb_count']})")
    print(f"mMTC切片分配: {best['Rm']} RB (并发用户数: {details['mmtc_count']})")
    print(f"总服务质量: {best['总服务质量']:.4f}\n")
    
    # 用户详细结果
    print("各用户详细结果:")
    print("-" * 100)
    print(f"{'用户':<8}{'切片':<10}{'排队时延':<12}{'传输时延':<12}{'总时延':<12}{'速率(Mbps)':<15}{'服务质量':<12}{'是否服务'}")
    print("-" * 100)
    
    user_types = {uid: df[df['用户编号'] == uid]['type'].values[0] for uid in df['用户编号']}
    sorted_users = sorted(df['用户编号'], key=lambda x: (user_types[x], x))
    
    for uid in sorted_users:
        utype = user_types[uid]
        res = details['user_results'][uid]
        # mMTC用户的服务质量在切片汇总中体现，这里显示接入状态
        qos = res['服务质量'] if utype != 'mMTC' else ('1.0000' if res['是否服务'] and not res['是否延迟超标'] else '0.0000')
        
        print(f"{uid:<8}{utype:<10}"
              f"{res['排队时延']:<12.2f}{res['传输时延']:<12.2f}{res['总时延']:<12.2f}"
              f"{res['速率(Mbps)']:<15.2f}{qos:<12}{'是' if res['是否服务'] else '否'}")
    
    print("-" * 100 + "\n")
    
    # 切片汇总
    print("各切片汇总:")
    print(f"URLLC: 平均服务质量 = {details['urllc_qos']/len(urllc_users):.4f}, 总服务质量 = {details['urllc_qos']:.4f}")
    print(f"eMBB: 平均服务质量 = {details['embb_qos']/len(embb_users):.4f}, 总服务质量 = {details['embb_qos']:.4f}")
    print(f"mMTC: 平均服务质量 = {details['mmtc_qos']:.4f}, 总服务质量 = {details['mmtc_qos']:.4f}\n")
    
    # Top 10方案
    print("Top 10最优资源分配方案:")
    print(f"{'RU':<5}{'Re':<5}{'Rm':<5}{'Total_QoS':<10}")
    for res in results[:10]:
        print(f"{res['RU']:<5}{res['Re']:<5}{res['Rm']:<5}{res['总服务质量']:.6f}")


# 主函数
if __name__ == "__main__":
    file_path = '/Users/a/Documents/华数/2025年第六届华数杯数学建模竞赛赛题/B题/附件/附件1/channel_data.xlsx'  # 替换为实际路径
    df = load_user_data(file_path)
    urllc_users = df[df['type'] == 'URLLC']
    embb_users = df[df['type'] == 'eMBB']
    allocation_results = enumerate_all_allocations(df)
    format_output(allocation_results, df)
