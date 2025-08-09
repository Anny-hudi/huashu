import pandas as pd
import math

# 系统参数
R_total = 50
power = 30
bandwidth_per_rb = 360e3
thermal_noise = -174
NF = 7

# SLA参数
URLLC_SLA_delay = 5
eMBB_SLA_delay = 100
eMBB_SLA_rate = 50
mMTC_SLA_delay = 500

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

# 加载数据
data = pd.read_excel('data1.xlsx')
user_data = data.iloc[0]

print("=== 详细调试信息 ===")
print(f"系统参数:")
print(f"  总资源块数: {R_total}")
print(f"  发射功率: {power} dBm")
print(f"  资源块带宽: {bandwidth_per_rb/1000:.1f} kHz")
print(f"  噪声系数: {NF}")
print(f"  热噪声: {thermal_noise} dBm/Hz")

print(f"\nSLA参数:")
print(f"  URLLC延迟限制: {URLLC_SLA_delay} ms")
print(f"  eMBB延迟限制: {eMBB_SLA_delay} ms")
print(f"  mMTC延迟限制: {mMTC_SLA_delay} ms")
print(f"  eMBB速率限制: {eMBB_SLA_rate} Mbps")

print(f"\n惩罚系数:")
print(f"  URLLC惩罚: {M_URLLC}")
print(f"  eMBB惩罚: {M_eMBB}")
print(f"  mMTC惩罚: {M_mMTC}")
print(f"  URLLC效用折扣: {alpha}")

print(f"\n用户信道增益:")
for key, value in user_data.items():
    if key != 'Time':
        print(f"  {key}: {value:.2f} dB")

# 测试最优方案的详细计算
print(f"\n=== 最优方案详细计算 ===")
urllc_rbs = 30
embb_rbs = 10
mmtc_rbs = 10

print(f"分配方案: URLLC({urllc_rbs}RB) + eMBB({embb_rbs}RB) + mMTC({mmtc_rbs}RB) = {urllc_rbs+embb_rbs+mmtc_rbs}RB")

# URLLC详细计算
print(f"\nURLLC切片详细计算 ({urllc_rbs} RB):")
for i in range(2):
    user_key = f'U{i+1}'
    channel_gain = user_data[user_key]
    sinr = calculate_sinr(power, channel_gain, urllc_rbs)
    rate = calculate_rate(sinr, urllc_rbs)
    data_size = 0.011
    delay = data_size / rate * 1000
    qos = calculate_urllc_qos(rate, delay)
    
    print(f"  {user_key}:")
    print(f"    信道增益: {channel_gain:.2f} dB")
    print(f"    接收功率: {10**((power - channel_gain)/10):.6f} mW")
    print(f"    噪声功率: {10**((thermal_noise + 10*math.log10(urllc_rbs * bandwidth_per_rb) + NF)/10):.6f} mW")
    print(f"    SINR: {sinr:.2f}")
    print(f"    传输速率: {rate:.2f} Mbps")
    print(f"    数据量: {data_size} Mbit")
    print(f"    延迟: {delay:.4f} ms")
    print(f"    QoS: {qos:.4f}")

# eMBB详细计算
print(f"\neMBB切片详细计算 ({embb_rbs} RB):")
for i in range(4):
    user_key = f'e{i+1}'
    channel_gain = user_data[user_key]
    sinr = calculate_sinr(power, channel_gain, embb_rbs)
    rate = calculate_rate(sinr, embb_rbs)
    data_size = 0.11
    delay = data_size / rate * 1000
    qos = calculate_embb_qos(rate, delay)
    
    print(f"  {user_key}:")
    print(f"    信道增益: {channel_gain:.2f} dB")
    print(f"    接收功率: {10**((power - channel_gain)/10):.6f} mW")
    print(f"    噪声功率: {10**((thermal_noise + 10*math.log10(embb_rbs * bandwidth_per_rb) + NF)/10):.6f} mW")
    print(f"    SINR: {sinr:.2f}")
    print(f"    传输速率: {rate:.2f} Mbps")
    print(f"    数据量: {data_size} Mbit")
    print(f"    延迟: {delay:.4f} ms")
    print(f"    QoS: {qos:.4f}")

# mMTC详细计算
print(f"\nmMTC切片详细计算 ({mmtc_rbs} RB):")
connected_users = 0
total_users = 0
for i in range(10):
    user_key = f'm{i+1}'
    channel_gain = user_data[user_key]
    sinr = calculate_sinr(power, channel_gain, mmtc_rbs)
    rate = calculate_rate(sinr, mmtc_rbs)
    connected = rate >= 1
    if connected:
        connected_users += 1
    total_users += 1
    
    print(f"  {user_key}:")
    print(f"    信道增益: {channel_gain:.2f} dB")
    print(f"    接收功率: {10**((power - channel_gain)/10):.6f} mW")
    print(f"    噪声功率: {10**((thermal_noise + 10*math.log10(mmtc_rbs * bandwidth_per_rb) + NF)/10):.6f} mW")
    print(f"    SINR: {sinr:.2f}")
    print(f"    传输速率: {rate:.2f} Mbps")
    print(f"    连接状态: {'连接' if connected else '未连接'}")

connection_ratio = connected_users / total_users
data_size = 0.013
avg_rate = mmtc_rbs * bandwidth_per_rb * math.log2(1 + 1) / 1e6
delay = data_size / avg_rate * 1000
qos = calculate_mmtc_qos(connection_ratio, delay)

print(f"\nmMTC整体统计:")
print(f"  连接用户数: {connected_users}")
print(f"  总用户数: {total_users}")
print(f"  连接率: {connection_ratio:.2f}")
print(f"  平均速率: {avg_rate:.2f} Mbps")
print(f"  数据量: {data_size} Mbit")
print(f"  延迟: {delay:.4f} ms")
print(f"  QoS: {qos:.4f}")

# 总QoS计算
total_qos = 0
urllc_qos = 0
embb_qos = 0
mmtc_qos = qos

for i in range(2):
    user_key = f'U{i+1}'
    channel_gain = user_data[user_key]
    sinr = calculate_sinr(power, channel_gain, urllc_rbs)
    rate = calculate_rate(sinr, urllc_rbs)
    data_size = 0.011
    delay = data_size / rate * 1000
    qos = calculate_urllc_qos(rate, delay)
    urllc_qos += qos

for i in range(4):
    user_key = f'e{i+1}'
    channel_gain = user_data[user_key]
    sinr = calculate_sinr(power, channel_gain, embb_rbs)
    rate = calculate_rate(sinr, embb_rbs)
    data_size = 0.11
    delay = data_size / rate * 1000
    qos = calculate_embb_qos(rate, delay)
    embb_qos += qos

total_qos = urllc_qos + embb_qos + mmtc_qos

print(f"\n=== 总QoS计算 ===")
print(f"URLLC总QoS: {urllc_qos:.4f}")
print(f"eMBB总QoS: {embb_qos:.4f}")
print(f"mMTC总QoS: {mmtc_qos:.4f}")
print(f"总QoS: {total_qos:.4f}") 