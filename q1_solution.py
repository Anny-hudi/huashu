import pandas as pd
import numpy as np
from itertools import product
import math

class NetworkSliceOptimizer:
    def __init__(self):
        # 系统参数
        self.R_total = 50  # 总资源块数
        self.power = 30  # 默认功率 dBm
        
        # 切片参数
        self.URLLC_MULTIPLE = 10  # URLLC资源块倍数
        self.eMBB_MULTIPLE = 5    # eMBB资源块倍数  
        self.mMTC_MULTIPLE = 2    # mMTC资源块倍数
        
        # SLA参数
        self.URLLC_SLA_delay = 5    # ms
        self.eMBB_SLA_delay = 100   # ms
        self.mMTC_SLA_delay = 500   # ms
        self.eMBB_SLA_rate = 50     # Mbps
        
        # 惩罚系数
        self.M_URLLC = 5
        self.M_eMBB = 3
        self.M_mMTC = 1
        
        # 效用折扣系数
        self.alpha = 0.95
        
        # 资源块参数
        self.bandwidth_per_rb = 360e3  # 360kHz
        self.time_slot = 1e-3          # 1ms
        
        # 噪声参数
        self.NF = 7  # 噪声系数
        self.thermal_noise = -174  # dBm/Hz
        
    def load_data(self, file_path):
        """加载数据"""
        data = pd.read_excel(file_path)
        return data.iloc[0]  # 取第一行数据
    
    def calculate_path_loss(self, distance):
        """计算路径损耗 (简化模型)"""
        # 使用简化的路径损耗模型
        return 128.1 + 37.6 * math.log10(distance)  # dB
    
    def calculate_channel_gain(self, user_data):
        """计算信道增益"""
        # 假设用户数据就是信道增益 (dB)
        return user_data
    
    def calculate_sinr(self, power_dbm, channel_gain_db, num_rbs, interference=0):
        """计算信干噪比"""
        # 转换功率单位
        power_mw = 10**((power_dbm - 30) / 10)
        
        # 计算接收功率
        channel_gain_linear = 10**(channel_gain_db / 10)
        received_power = power_mw * channel_gain_linear
        
        # 计算噪声功率
        noise_power = 10**((self.thermal_noise + 10*math.log10(num_rbs * self.bandwidth_per_rb) + self.NF) / 10)
        
        # 计算SINR
        sinr = received_power / (interference + noise_power)
        return sinr
    
    def calculate_transmission_rate(self, sinr, num_rbs):
        """计算传输速率"""
        rate = num_rbs * self.bandwidth_per_rb * math.log2(1 + sinr)
        return rate / 1e6  # 转换为Mbps
    
    def calculate_urllc_qos(self, rate, delay):
        """计算URLLC服务质量"""
        if delay <= self.URLLC_SLA_delay:
            return self.alpha ** delay
        else:
            return -self.M_URLLC
    
    def calculate_embb_qos(self, rate, delay):
        """计算eMBB服务质量"""
        if delay <= self.eMBB_SLA_delay:
            if rate >= self.eMBB_SLA_rate:
                return 1.0
            else:
                return rate / self.eMBB_SLA_rate
        else:
            return -self.M_eMBB
    
    def calculate_mmtc_qos(self, connection_ratio, delay):
        """计算mMTC服务质量"""
        if delay <= self.mMTC_SLA_delay:
            return connection_ratio
        else:
            return -self.M_mMTC
    
    def evaluate_allocation(self, urllc_rbs, embb_rbs, mmtc_rbs, user_data):
        """评估资源分配方案的服务质量"""
        total_qos = 0
        
        # URLLC用户评估
        if urllc_rbs > 0:
            for i in range(2):  # U1, U2
                user_key = f'U{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = self.calculate_sinr(self.power, channel_gain, urllc_rbs)
                    rate = self.calculate_transmission_rate(sinr, urllc_rbs)
                    
                    # 估算延迟 (简化模型)
                    data_size = 0.011  # 平均数据量 0.01-0.012 Mbit
                    delay = data_size / rate * 1000  # 转换为ms
                    
                    qos = self.calculate_urllc_qos(rate, delay)
                    total_qos += qos
        
        # eMBB用户评估
        if embb_rbs > 0:
            for i in range(4):  # e1, e2, e3, e4
                user_key = f'e{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = self.calculate_sinr(self.power, channel_gain, embb_rbs)
                    rate = self.calculate_transmission_rate(sinr, embb_rbs)
                    
                    # 估算延迟
                    data_size = 0.11  # 平均数据量 0.1-0.12 Mbit
                    delay = data_size / rate * 1000
                    
                    qos = self.calculate_embb_qos(rate, delay)
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
                    sinr = self.calculate_sinr(self.power, channel_gain, mmtc_rbs)
                    rate = self.calculate_transmission_rate(sinr, mmtc_rbs)
                    
                    # 简化的连接判断
                    if rate >= 1:  # 1Mbps SLA
                        connected_users += 1
            
            connection_ratio = connected_users / total_users if total_users > 0 else 0
            
            # 估算延迟
            data_size = 0.013  # 平均数据量 0.012-0.014 Mbit
            avg_rate = mmtc_rbs * self.bandwidth_per_rb * math.log2(1 + 1) / 1e6  # 简化计算
            delay = data_size / avg_rate * 1000
            
            qos = self.calculate_mmtc_qos(connection_ratio, delay)
            total_qos += qos
        
        return total_qos
    
    def optimize_allocation(self, user_data):
        """优化资源分配"""
        best_qos = float('-inf')
        best_allocation = None
        
        # 生成所有可能的分配方案
        # 考虑倍数约束
        urllc_possible = [0, 10, 20, 30, 40, 50]  # 必须是10的倍数
        embb_possible = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # 必须是5的倍数
        mmtc_possible = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]  # 必须是2的倍数
        
        for urllc_rbs in urllc_possible:
            for embb_rbs in embb_possible:
                for mmtc_rbs in mmtc_possible:
                    # 检查资源约束
                    if urllc_rbs + embb_rbs + mmtc_rbs <= self.R_total:
                        qos = self.evaluate_allocation(urllc_rbs, embb_rbs, mmtc_rbs, user_data)
                        
                        if qos > best_qos:
                            best_qos = qos
                            best_allocation = (urllc_rbs, embb_rbs, mmtc_rbs)
        
        return best_allocation, best_qos
    
    def solve_problem_1(self, data_file):
        """解决第一题"""
        print("=== 第一题：微基站资源块分配优化 ===")
        
        # 加载数据
        user_data = self.load_data(data_file)
        print(f"用户数据: {dict(user_data)}")
        
        # 优化分配
        best_allocation, best_qos = self.optimize_allocation(user_data)
        
        print(f"\n最优资源分配方案:")
        print(f"URLLC切片: {best_allocation[0]} 个资源块")
        print(f"eMBB切片: {best_allocation[1]} 个资源块") 
        print(f"mMTC切片: {best_allocation[2]} 个资源块")
        print(f"总服务质量: {best_qos:.4f}")
        
        # 详细分析
        print(f"\n详细分析:")
        urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
        
        if urllc_rbs > 0:
            print(f"URLLC切片 ({urllc_rbs} RB):")
            for i in range(2):
                user_key = f'U{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = self.calculate_sinr(self.power, channel_gain, urllc_rbs)
                    rate = self.calculate_transmission_rate(sinr, urllc_rbs)
                    print(f"  {user_key}: 速率={rate:.2f} Mbps, 信道增益={channel_gain:.2f} dB")
        
        if embb_rbs > 0:
            print(f"eMBB切片 ({embb_rbs} RB):")
            for i in range(4):
                user_key = f'e{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = self.calculate_sinr(self.power, channel_gain, embb_rbs)
                    rate = self.calculate_transmission_rate(sinr, embb_rbs)
                    print(f"  {user_key}: 速率={rate:.2f} Mbps, 信道增益={channel_gain:.2f} dB")
        
        if mmtc_rbs > 0:
            print(f"mMTC切片 ({mmtc_rbs} RB):")
            connected = 0
            for i in range(10):
                user_key = f'm{i+1}'
                if user_key in user_data:
                    channel_gain = user_data[user_key]
                    sinr = self.calculate_sinr(self.power, channel_gain, mmtc_rbs)
                    rate = self.calculate_transmission_rate(sinr, mmtc_rbs)
                    status = "连接" if rate >= 1 else "未连接"
                    if rate >= 1:
                        connected += 1
                    print(f"  {user_key}: 速率={rate:.2f} Mbps, 状态={status}")
            print(f"  连接率: {connected}/10 = {connected/10:.2f}")
        
        return best_allocation, best_qos

if __name__ == "__main__":
    optimizer = NetworkSliceOptimizer()
    best_allocation, best_qos = optimizer.solve_problem_1('data1.xlsx') 