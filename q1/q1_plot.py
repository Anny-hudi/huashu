import pandas as pd
import numpy as np
import math
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置seaborn中文字体
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['font.size'] = 12

# 额外的字体设置选项
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

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
    
    # 初始化user_data变量
    user_data = {
        'large_scale': {},
        'small_scale': {},
        'task_flow': {},
        'user_pos': {}
    }
    
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
    
    def plot_histograms(user_data):
        """绘制用户数据的联合分布图（中间散点图，外围边际直方图）"""
        # 设置图表样式和字体
        sns.set(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 高级莫兰迪配色方案
        morandi_colors = {
            'primary': ['#8B7355', '#A67B5B', '#C68E17', '#DAA520'],  # 暖棕色系
            'secondary': ['#6B8E23', '#556B2F', '#8FBC8F', '#90EE90'],  # 柔和绿色系
            'accent': ['#CD853F', '#DEB887', '#F5DEB3', '#FFE4B5'],  # 米色系
            'neutral': ['#F5F5DC', '#FAEBD7', '#FFEFD5', '#FFF8DC']  # 浅色系
        }
        
        # 1. 大规模衰减联合分布图
        large_scale_values = list(user_data['large_scale'].values())
        large_scale_indices = range(len(large_scale_values))
        
        fig1 = plt.figure(figsize=(8, 6))
        g1 = sns.JointGrid(x=large_scale_indices, y=large_scale_values, height=6, ratio=3)
        g1.plot_joint(sns.scatterplot, color=morandi_colors['primary'][0], alpha=0.7, s=60)
        g1.plot_marginals(sns.histplot, color=morandi_colors['primary'][0], alpha=0.6, 
                         edgecolor=morandi_colors['accent'][2], linewidth=1)
        g1.ax_joint.set_xlabel('用户索引', fontsize=12, color=morandi_colors['primary'][0])
        g1.ax_joint.set_ylabel('大规模衰减 (dB)', fontsize=12, color=morandi_colors['primary'][0])
        g1.ax_joint.set_title('大规模衰减分布', fontsize=14, fontweight='bold', color=morandi_colors['primary'][0], pad=20)
        g1.ax_joint.grid(True, alpha=0.3, color=morandi_colors['accent'][1])
        g1.ax_joint.set_facecolor(morandi_colors['neutral'][3])
        g1.fig.patch.set_facecolor(morandi_colors['neutral'][2])
        
        # 保存第一个图
        save_path1 = '/Users/a/Documents/Projects/web_question/q1/large_scale_joint_plot.png'
        g1.fig.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor=morandi_colors['neutral'][2])
        print(f"大规模衰减联合分布图已保存到: {save_path1}")
        plt.close(fig1)
        
        # 2. 小规模瑞丽衰减联合分布图
        small_scale_values = list(user_data['small_scale'].values())
        small_scale_indices = range(len(small_scale_values))
        
        fig2 = plt.figure(figsize=(8, 6))
        g2 = sns.JointGrid(x=small_scale_indices, y=small_scale_values, height=6, ratio=3)
        g2.plot_joint(sns.scatterplot, color=morandi_colors['secondary'][0], alpha=0.7, s=60)
        g2.plot_marginals(sns.histplot, color=morandi_colors['secondary'][0], alpha=0.6,
                         edgecolor=morandi_colors['accent'][2], linewidth=1)
        g2.ax_joint.set_xlabel('用户索引', fontsize=12, color=morandi_colors['primary'][0])
        g2.ax_joint.set_ylabel('小规模瑞丽衰减', fontsize=12, color=morandi_colors['primary'][0])
        g2.ax_joint.set_title('小规模瑞丽衰减分布', fontsize=14, fontweight='bold', color=morandi_colors['primary'][0], pad=20)
        g2.ax_joint.grid(True, alpha=0.3, color=morandi_colors['accent'][1])
        g2.ax_joint.set_facecolor(morandi_colors['neutral'][3])
        g2.fig.patch.set_facecolor(morandi_colors['neutral'][2])
        
        # 保存第二个图
        save_path2 = '/Users/a/Documents/Projects/web_question/q1/small_scale_joint_plot.png'
        g2.fig.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor=morandi_colors['neutral'][2])
        print(f"小规模瑞丽衰减联合分布图已保存到: {save_path2}")
        plt.close(fig2)
        
        # 3. 任务流联合分布图
        task_flow_values = list(user_data['task_flow'].values())
        task_flow_indices = range(len(task_flow_values))
        
        fig3 = plt.figure(figsize=(8, 6))
        g3 = sns.JointGrid(x=task_flow_indices, y=task_flow_values, height=6, ratio=3)
        g3.plot_joint(sns.scatterplot, color=morandi_colors['accent'][0], alpha=0.7, s=60)
        g3.plot_marginals(sns.histplot, color=morandi_colors['accent'][0], alpha=0.6,
                         edgecolor=morandi_colors['accent'][2], linewidth=1)
        g3.ax_joint.set_xlabel('用户索引', fontsize=12, color=morandi_colors['primary'][0])
        g3.ax_joint.set_ylabel('任务流 (Mbit)', fontsize=12, color=morandi_colors['primary'][0])
        g3.ax_joint.set_title('任务流分布', fontsize=14, fontweight='bold', color=morandi_colors['primary'][0], pad=20)
        g3.ax_joint.grid(True, alpha=0.3, color=morandi_colors['accent'][1])
        g3.ax_joint.set_facecolor(morandi_colors['neutral'][3])
        g3.fig.patch.set_facecolor(morandi_colors['neutral'][2])
        
        # 保存第三个图
        save_path3 = '/Users/a/Documents/Projects/web_question/q1/task_flow_joint_plot.png'
        g3.fig.savefig(save_path3, dpi=300, bbox_inches='tight', facecolor=morandi_colors['neutral'][2])
        print(f"任务流联合分布图已保存到: {save_path3}")
        plt.close(fig3)
        
        print("所有三个联合分布图已分别保存完成！")
    
    def plot_user_analysis(user_data):
        """绘制用户数据分析图表"""
        # 设置图表样式和字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 高级莫兰迪配色方案
        morandi_colors = {
            'primary': ['#8B7355', '#A67B5B', '#C68E17', '#DAA520'],  # 暖棕色系
            'secondary': ['#6B8E23', '#556B2F', '#8FBC8F', '#90EE90'],  # 柔和绿色系
            'accent': ['#CD853F', '#DEB887', '#F5DEB3', '#FFE4B5'],  # 米色系
            'neutral': ['#F5F5DC', '#FAEBD7', '#FFEFD5', '#FFF8DC']  # 浅色系
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('用户数据分析', fontsize=16, fontweight='bold', color=morandi_colors['primary'][0])
        
        # 1. 用户位置散点图
        ax1 = axes[0, 0]
        user_positions = []
        user_labels = []
        for user in ['U1', 'U2', 'e1', 'e2', 'e3', 'e4', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']:
            x = user_data['user_pos'][f'{user}_X']
            y = user_data['user_pos'][f'{user}_Y']
            user_positions.append([x, y])
            user_labels.append(user)
        
        user_positions = np.array(user_positions)
        ax1.scatter(user_positions[:, 0], user_positions[:, 1], 
                   c=morandi_colors['primary'][0], alpha=0.8, s=120, edgecolors=morandi_colors['accent'][2], linewidth=1)
        for i, label in enumerate(user_labels):
            ax1.annotate(label, (user_positions[i, 0], user_positions[i, 1]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=morandi_colors['neutral'][0], alpha=0.8))
        ax1.set_xlabel('X坐标 (m)', fontsize=12, color=morandi_colors['primary'][0])
        ax1.set_ylabel('Y坐标 (m)', fontsize=12, color=morandi_colors['primary'][0])
        ax1.set_title('用户位置分布', fontsize=14, fontweight='bold', color=morandi_colors['primary'][0])
        ax1.grid(True, alpha=0.3, color=morandi_colors['accent'][1])
        ax1.set_facecolor(morandi_colors['neutral'][3])
        
        # 2. 任务流数据条形图
        ax2 = axes[0, 1]
        users = ['U1', 'U2', 'e1', 'e2', 'e3', 'e4', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
        task_sizes = [user_data['task_flow'][user] for user in users]
        # 使用莫兰迪配色
        colors = [morandi_colors['primary'][0] if 'U' in user else 
                 morandi_colors['secondary'][0] if 'e' in user else 
                 morandi_colors['accent'][0] for user in users]
        
        bars = ax2.bar(range(len(users)), task_sizes, color=colors, alpha=0.8, 
                       edgecolor=morandi_colors['accent'][2], linewidth=1)
        ax2.set_xlabel('用户', fontsize=12, color=morandi_colors['primary'][0])
        ax2.set_ylabel('任务数据量 (Mbit)', fontsize=12, color=morandi_colors['primary'][0])
        ax2.set_title('用户任务流数据', fontsize=14, fontweight='bold', color=morandi_colors['primary'][0])
        ax2.set_xticks(range(len(users)))
        ax2.set_xticklabels(users, rotation=45, ha='right', color=morandi_colors['primary'][0])
        ax2.grid(True, alpha=0.3, color=morandi_colors['accent'][1])
        ax2.set_facecolor(morandi_colors['neutral'][3])
        
        # 3. 大规模衰减热力图
        ax3 = axes[1, 0]
        large_scale_values = [user_data['large_scale'][user] for user in users]
        small_scale_values = [user_data['small_scale'][user] for user in users]
        
        # 创建衰减数据矩阵
        attenuation_data = np.array([large_scale_values, small_scale_values])
        im = ax3.imshow(attenuation_data, cmap='viridis', aspect='auto')
        ax3.set_xlabel('用户', fontsize=12, color=morandi_colors['primary'][0])
        ax3.set_ylabel('衰减类型', fontsize=12, color=morandi_colors['primary'][0])
        ax3.set_title('用户衰减特性', fontsize=14, fontweight='bold', color=morandi_colors['primary'][0])
        ax3.set_xticks(range(len(users)))
        ax3.set_xticklabels(users, rotation=45, ha='right', color=morandi_colors['primary'][0])
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['大规模衰减', '小规模瑞丽衰减'], color=morandi_colors['primary'][0])
        ax3.set_facecolor(morandi_colors['neutral'][3])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('衰减值', fontsize=12, color=morandi_colors['primary'][0])
        
        # 4. 用户类型统计饼图
        ax4 = axes[1, 1]
        user_types = ['URLLC', 'eMBB', 'mMTC']
        user_counts = [2, 4, 10]  # 根据代码中的用户数量
        colors_pie = [morandi_colors['primary'][0], morandi_colors['secondary'][0], morandi_colors['accent'][0]]
        
        wedges, texts, autotexts = ax4.pie(user_counts, labels=user_types, colors=colors_pie, 
                                           autopct='%1.1f%%', startangle=90, 
                                           textprops={'color': morandi_colors['primary'][0]})
        ax4.set_title('用户类型分布', fontsize=14, fontweight='bold', color=morandi_colors['primary'][0])
        
        # 设置饼图文字样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 设置整体背景色
        fig.patch.set_facecolor(morandi_colors['neutral'][2])
        
        plt.tight_layout()
        
        # 保存图表到指定目录
        save_path = '/Users/a/Documents/Projects/web_question/q1/user_analysis_chart.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=morandi_colors['neutral'][2])
        print(f"用户数据分析图表已保存到: {save_path}")
        
        plt.show()
    
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
    
    # 在函数内部绘制直方图，确保能访问到正确的user_data
    print(f"\n=== 绘制用户数据分布直方图 ===")
    plot_histograms(user_data)
    
    # 绘制额外的用户数据分析图表
    print(f"\n=== 绘制用户数据分析图表 ===")
    plot_user_analysis(user_data)

if __name__ == "__main__":
    solve_problem_1_fixed_priority() 