import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 如果上述字体不可用，尝试使用系统默认字体
try:
    # 测试中文字体是否可用
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, '测试中文', fontsize=12)
    plt.close(fig)
except:
    # 如果中文字体不可用，使用英文标签
    print("警告: 中文字体不可用，将使用英文标签")
    USE_CHINESE = False
else:
    USE_CHINESE = True


def get_label(chinese: str, english: str) -> str:
    """根据字体可用性返回中文或英文标签"""
    return chinese if USE_CHINESE else english


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def detect_base_stations(df: pd.DataFrame):
    bs_list = []
    for col in df.columns:
        if "_URLLC_RB" in col:
            bs = col.split("_URLLC_RB")[0]
            bs_list.append(bs)
    return sorted(list(set(bs_list)))


def plot_reward(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(10, 6))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 奖励变化
    if USE_CHINESE:
        ax1.plot(df.index, df["reward"], marker='o', linewidth=1, color='blue', label='即时奖励')
        ax1.plot(df.index, df["total_reward"], marker='s', linewidth=1, color='red', label='累积奖励')
        ax1.set_title("奖励变化趋势")
        ax1.set_xlabel("决策索引")
        ax1.set_ylabel("奖励值")
    else:
        ax1.plot(df.index, df["reward"], marker='o', linewidth=1, color='blue', label='Instant Reward')
        ax1.plot(df.index, df["total_reward"], marker='s', linewidth=1, color='red', label='Cumulative Reward')
        ax1.set_title("Reward Trend")
        ax1.set_xlabel("Decision Index")
        ax1.set_ylabel("Reward Value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 奖励分布
    ax2.hist(df["reward"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    if USE_CHINESE:
        ax2.set_title("奖励分布直方图")
        ax2.set_xlabel("奖励值")
        ax2.set_ylabel("频次")
    else:
        ax2.set_title("Reward Distribution Histogram")
        ax2.set_xlabel("Reward Value")
        ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_over_time.png"), dpi=150)
    plt.close()


def plot_rb_per_bs(df: pd.DataFrame, bs: str, outdir: str):
    u = df[f"{bs}_URLLC_RB"]
    e = df[f"{bs}_eMBB_RB"]
    m = df[f"{bs}_mMTC_RB"]

    plt.figure(figsize=(10, 6))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 资源分配堆叠图
    ax1.stackplot(df.index, u, e, m, labels=["URLLC", "eMBB", "mMTC"], alpha=0.8)
    if USE_CHINESE:
        ax1.set_title(f"{bs} RB分配策略")
        ax1.set_xlabel("决策索引")
        ax1.set_ylabel("RB数量")
    else:
        ax1.set_title(f"{bs} RB Allocation Strategy")
        ax1.set_xlabel("Decision Index")
        ax1.set_ylabel("RB Count")
    ax1.legend(loc="upper right", ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 资源利用率
    total_rb = u + e + m
    utilization = total_rb / 50.0 * 100  # 假设总RB为50
    ax2.plot(df.index, utilization, marker='o', linewidth=1, color='green')
    if USE_CHINESE:
        ax2.set_title(f"{bs} 资源利用率")
        ax2.set_xlabel("决策索引")
        ax2.set_ylabel("利用率 (%)")
    else:
        ax2.set_title(f"{bs} Resource Utilization")
        ax2.set_xlabel("Decision Index")
        ax2.set_ylabel("Utilization (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{bs}_rb_allocation.png"), dpi=150)
    plt.close()


def plot_power_per_bs(df: pd.DataFrame, bs: str, outdir: str):
    pu = df[f"{bs}_URLLC_P"]
    pe = df[f"{bs}_eMBB_P"]
    pm = df[f"{bs}_mMTC_P"]

    plt.figure(figsize=(10, 6))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 功率变化
    ax1.plot(df.index, pu, label="URLLC", marker='o', linewidth=1, color='red')
    ax1.plot(df.index, pe, label="eMBB", marker='o', linewidth=1, color='blue')
    ax1.plot(df.index, pm, label="mMTC", marker='o', linewidth=1, color='green')
    if USE_CHINESE:
        ax1.set_title(f"{bs} 功率控制策略")
        ax1.set_xlabel("决策索引")
        ax1.set_ylabel("功率 (dBm)")
    else:
        ax1.set_title(f"{bs} Power Control Strategy")
        ax1.set_xlabel("Decision Index")
        ax1.set_ylabel("Power (dBm)")
    ax1.set_ylim(0, 40)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", ncol=3, fontsize=8)
    
    # 总功率消耗
    total_power = pu + pe + pm
    ax2.plot(df.index, total_power, marker='s', linewidth=1, color='orange')
    if USE_CHINESE:
        ax2.set_title(f"{bs} 总功率消耗")
        ax2.set_xlabel("决策索引")
        ax2.set_ylabel("总功率 (dBm)")
    else:
        ax2.set_title(f"{bs} Total Power Consumption")
        ax2.set_xlabel("Decision Index")
        ax2.set_ylabel("Total Power (dBm)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{bs}_power_levels.png"), dpi=150)
    plt.close()


def plot_timeout_penalties(df: pd.DataFrame, outdir: str):
    """绘制超时惩罚分析图"""
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    slice_names = ["URLLC", "eMBB", "mMTC"]
    colors = ['red', 'blue', 'green']
    
    # 超时惩罚趋势
    for i, slice_name in enumerate(slice_names):
        col = f"{slice_name}_timeout_penalty"
        if col in df.columns:
            ax1.plot(df.index, df[col], marker='o', linewidth=1, 
                    color=colors[i], label=slice_name)
    ax1.set_title(get_label("超时惩罚变化趋势", "Timeout Penalty Trend"))
    ax1.set_xlabel(get_label("决策索引", "Decision Index"))
    ax1.set_ylabel(get_label("超时惩罚次数", "Timeout Penalty Count"))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 超时惩罚分布
    for i, slice_name in enumerate(slice_names):
        col = f"{slice_name}_timeout_penalty"
        if col in df.columns:
            ax2.hist(df[col], bins=10, alpha=0.6, color=colors[i], 
                    label=slice_name, edgecolor='black')
    ax2.set_title(get_label("超时惩罚分布", "Timeout Penalty Distribution"))
    ax2.set_xlabel(get_label("超时惩罚次数", "Timeout Penalty Count"))
    ax2.set_ylabel(get_label("频次", "Frequency"))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 累积超时惩罚
    cumulative_penalties = {}
    for slice_name in slice_names:
        col = f"{slice_name}_timeout_penalty"
        if col in df.columns:
            cumulative_penalties[slice_name] = df[col].cumsum()
    
    for i, slice_name in enumerate(slice_names):
        if slice_name in cumulative_penalties:
            ax3.plot(df.index, cumulative_penalties[slice_name], 
                    marker='s', linewidth=1, color=colors[i], label=slice_name)
    ax3.set_title(get_label("累积超时惩罚", "Cumulative Timeout Penalty"))
    ax3.set_xlabel(get_label("决策索引", "Decision Index"))
    ax3.set_ylabel(get_label("累积惩罚次数", "Cumulative Penalty Count"))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 超时惩罚热力图
    penalty_matrix = []
    for slice_name in slice_names:
        col = f"{slice_name}_timeout_penalty"
        if col in df.columns:
            penalty_matrix.append(df[col].values)
    
    if penalty_matrix:
        im = ax4.imshow(penalty_matrix, cmap='Reds', aspect='auto')
        ax4.set_title(get_label("超时惩罚热力图", "Timeout Penalty Heatmap"))
        ax4.set_xlabel(get_label("决策索引", "Decision Index"))
        ax4.set_ylabel(get_label("切片类型", "Slice Type"))
        ax4.set_yticks(range(len(slice_names)))
        ax4.set_yticklabels(slice_names)
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "timeout_penalties.png"), dpi=150)
    plt.close()


def plot_delay_analysis(df: pd.DataFrame, outdir: str):
    """绘制时延变化分析图"""
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
    
    slice_names = ["URLLC", "eMBB", "mMTC"]
    colors = ['red', 'blue', 'green']
    
    # 平均时延变化
    for i, slice_name in enumerate(slice_names):
        col = f"{slice_name}_avg_delay"
        if col in df.columns:
            ax1.plot(df.index, df[col], marker='o', linewidth=1, 
                    color=colors[i], label=slice_name)
    ax1.set_title(get_label("平均时延变化趋势", "Average Delay Trend"))
    ax1.set_xlabel(get_label("决策索引", "Decision Index"))
    ax1.set_ylabel(get_label("平均时延 (ms)", "Average Delay (ms)"))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 最大时延变化
    for i, slice_name in enumerate(slice_names):
        col = f"{slice_name}_max_delay"
        if col in df.columns:
            ax2.plot(df.index, df[col], marker='s', linewidth=1, 
                    color=colors[i], label=slice_name)
    ax2.set_title(get_label("最大时延变化趋势", "Maximum Delay Trend"))
    ax2.set_xlabel(get_label("决策索引", "Decision Index"))
    ax2.set_ylabel(get_label("最大时延 (ms)", "Maximum Delay (ms)"))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 时延方差变化
    for i, slice_name in enumerate(slice_names):
        col = f"{slice_name}_delay_variance"
        if col in df.columns:
            ax3.plot(df.index, df[col], marker='^', linewidth=1, 
                    color=colors[i], label=slice_name)
    ax3.set_title(get_label("时延方差变化趋势", "Delay Variance Trend"))
    ax3.set_xlabel(get_label("决策索引", "Decision Index"))
    ax3.set_ylabel(get_label("时延方差", "Delay Variance"))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 时延分布箱线图
    delay_data = []
    labels = []
    for slice_name in slice_names:
        col = f"{slice_name}_avg_delay"
        if col in df.columns:
            delay_data.append(df[col].values)
            labels.append(slice_name)
    
    if delay_data:
        ax4.boxplot(delay_data, labels=labels, patch_artist=True)
        ax4.set_title(get_label("时延分布箱线图", "Delay Distribution Boxplot"))
        ax4.set_ylabel(get_label("时延 (ms)", "Delay (ms)"))
        ax4.grid(True, alpha=0.3)
    
    # 时延与SLA对比
    sla_values = {"URLLC": 5, "eMBB": 100, "mMTC": 500}
    for i, slice_name in enumerate(slice_names):
        col = f"{slice_name}_avg_delay"
        if col in df.columns:
            if USE_CHINESE:
                ax5.plot(df.index, df[col], marker='o', linewidth=1, 
                        color=colors[i], label=f"{slice_name} 实际时延")
                ax5.axhline(y=sla_values[slice_name], color=colors[i], 
                           linestyle='--', alpha=0.7, label=f"{slice_name} SLA")
            else:
                ax5.plot(df.index, df[col], marker='o', linewidth=1, 
                        color=colors[i], label=f"{slice_name} Actual Delay")
                ax5.axhline(y=sla_values[slice_name], color=colors[i], 
                           linestyle='--', alpha=0.7, label=f"{slice_name} SLA")
    ax5.set_title(get_label("时延与SLA对比", "Delay vs SLA Comparison"))
    ax5.set_xlabel(get_label("决策索引", "Decision Index"))
    ax5.set_ylabel(get_label("时延 (ms)", "Delay (ms)"))
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 时延热力图
    delay_matrix = []
    for slice_name in slice_names:
        col = f"{slice_name}_avg_delay"
        if col in df.columns:
            delay_matrix.append(df[col].values)
    
    if delay_matrix:
        im = ax6.imshow(delay_matrix, cmap='viridis', aspect='auto')
        ax6.set_title(get_label("时延热力图", "Delay Heatmap"))
        ax6.set_xlabel(get_label("决策索引", "Decision Index"))
        ax6.set_ylabel(get_label("切片类型", "Slice Type"))
        ax6.set_yticks(range(len(slice_names)))
        ax6.set_yticklabels(slice_names)
        plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "delay_analysis.png"), dpi=150)
    plt.close()


def plot_markov_decision_analysis(df: pd.DataFrame, outdir: str):
    """绘制马尔可夫决策分析图"""
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 状态转移分析 - 资源分配变化
    bs_list = detect_base_stations(df)
    if len(bs_list) > 0:
        bs = bs_list[0]  # 选择第一个基站进行分析
        
        # 计算资源分配变化率
        urllc_rb = df[f"{bs}_URLLC_RB"].values
        embb_rb = df[f"{bs}_eMBB_RB"].values
        mmtc_rb = df[f"{bs}_mMTC_RB"].values
        
        urllc_change = np.diff(urllc_rb)
        embb_change = np.diff(embb_rb)
        mmtc_change = np.diff(mmtc_rb)
        
        ax1.plot(range(1, len(urllc_change)+1), urllc_change, 
                marker='o', label='URLLC', color='red')
        ax1.plot(range(1, len(embb_change)+1), embb_change, 
                marker='s', label='eMBB', color='blue')
        ax1.plot(range(1, len(mmtc_change)+1), mmtc_change, 
                marker='^', label='mMTC', color='green')
        ax1.set_title(f"{bs} {get_label('资源分配变化率', 'Resource Allocation Change Rate')}")
        ax1.set_xlabel(get_label("决策索引", "Decision Index"))
        ax1.set_ylabel(get_label("RB变化量", "RB Change Amount"))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 决策稳定性分析
    if len(bs_list) > 0:
        bs = bs_list[0]
        # 计算决策稳定性（连续相同决策的比例）
        urllc_rb = df[f"{bs}_URLLC_RB"].values
        embb_rb = df[f"{bs}_eMBB_RB"].values
        mmtc_rb = df[f"{bs}_mMTC_RB"].values
        
        stability_urllc = np.sum(np.diff(urllc_rb) == 0) / max(1, len(urllc_rb) - 1)
        stability_embb = np.sum(np.diff(embb_rb) == 0) / max(1, len(embb_rb) - 1)
        stability_mmtc = np.sum(np.diff(mmtc_rb) == 0) / max(1, len(mmtc_rb) - 1)
        
        slices = ['URLLC', 'eMBB', 'mMTC']
        stabilities = [stability_urllc, stability_embb, stability_mmtc]
        colors = ['red', 'blue', 'green']
        
        bars = ax2.bar(slices, stabilities, color=colors, alpha=0.7)
        ax2.set_title(f"{bs} {get_label('决策稳定性分析', 'Decision Stability Analysis')}")
        ax2.set_ylabel(get_label("稳定性比例", "Stability Ratio"))
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars, stabilities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom')
    
    # 状态空间探索
    if len(bs_list) > 0:
        bs = bs_list[0]
        urllc_rb = df[f"{bs}_URLLC_RB"].values
        embb_rb = df[f"{bs}_eMBB_RB"].values
        
        # 创建状态空间散点图
        ax3.scatter(urllc_rb, embb_rb, c=df.index, cmap='viridis', alpha=0.7)
        ax3.set_title(f"{bs} {get_label('状态空间探索', 'State Space Exploration')}")
        ax3.set_xlabel("URLLC RB")
        ax3.set_ylabel("eMBB RB")
        ax3.grid(True, alpha=0.3)
        
        # 添加颜色条
        scatter = ax3.scatter(urllc_rb, embb_rb, c=df.index, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax3, label=get_label('决策索引', 'Decision Index'))
    
    # 决策序列分析
    if len(bs_list) > 0:
        bs = bs_list[0]
        urllc_rb = df[f"{bs}_URLLC_RB"].values
        embb_rb = df[f"{bs}_eMBB_RB"].values
        mmtc_rb = df[f"{bs}_mMTC_RB"].values
        
        # 创建决策序列热力图
        decision_matrix = np.vstack([urllc_rb, embb_rb, mmtc_rb])
        im = ax4.imshow(decision_matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_title(f"{bs} {get_label('决策序列热力图', 'Decision Sequence Heatmap')}")
        ax4.set_xlabel(get_label("决策索引", "Decision Index"))
        ax4.set_ylabel(get_label("切片类型", "Slice Type"))
        ax4.set_yticks(range(3))
        ax4.set_yticklabels(['URLLC', 'eMBB', 'mMTC'])
        plt.colorbar(im, ax=ax4, label=get_label('RB数量', 'RB Count'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "markov_decision_analysis.png"), dpi=150)
    plt.close()


def plot_queue_analysis(df: pd.DataFrame, outdir: str):
    """绘制队列分析图"""
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    bs_list = detect_base_stations(df)
    slice_names = ["URLLC", "eMBB", "mMTC"]
    colors = ['red', 'blue', 'green']
    
    # 队列长度变化
    for i, bs in enumerate(bs_list):
        for j, slice_name in enumerate(slice_names):
            col = f"{bs}_{slice_name}_queue_len"
            if col in df.columns:
                ax1.plot(df.index, df[col], marker='o', linewidth=1, 
                        color=colors[j], alpha=0.7, 
                        label=f"{bs}_{slice_name}")
    ax1.set_title(get_label("队列长度变化趋势", "Queue Length Trend"))
    ax1.set_xlabel(get_label("决策索引", "Decision Index"))
    ax1.set_ylabel(get_label("队列长度", "Queue Length"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 平均排队时延
    for i, bs in enumerate(bs_list):
        for j, slice_name in enumerate(slice_names):
            col = f"{bs}_{slice_name}_avg_queue_delay"
            if col in df.columns:
                ax2.plot(df.index, df[col], marker='s', linewidth=1, 
                        color=colors[j], alpha=0.7, 
                        label=f"{bs}_{slice_name}")
    ax2.set_title(get_label("平均排队时延变化", "Average Queue Delay Trend"))
    ax2.set_xlabel(get_label("决策索引", "Decision Index"))
    ax2.set_ylabel(get_label("平均排队时延 (ms)", "Average Queue Delay (ms)"))
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 队列长度分布
    queue_data = []
    labels = []
    for bs in bs_list:
        for slice_name in slice_names:
            col = f"{bs}_{slice_name}_queue_len"
            if col in df.columns:
                queue_data.append(df[col].values)
                labels.append(f"{bs}_{slice_name}")
    
    if queue_data:
        ax3.boxplot(queue_data, labels=labels)
        ax3.set_title(get_label("队列长度分布", "Queue Length Distribution"))
        ax3.set_ylabel(get_label("队列长度", "Queue Length"))
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # 队列热力图
    queue_matrix = []
    for bs in bs_list:
        for slice_name in slice_names:
            col = f"{bs}_{slice_name}_queue_len"
            if col in df.columns:
                queue_matrix.append(df[col].values)
    
    if queue_matrix:
        im = ax4.imshow(queue_matrix, cmap='Blues', aspect='auto')
        ax4.set_title(get_label("队列长度热力图", "Queue Length Heatmap"))
        ax4.set_xlabel(get_label("决策索引", "Decision Index"))
        ax4.set_ylabel(get_label("基站-切片组合", "BS-Slice Combination"))
        ax4.set_yticks(range(len(queue_matrix)))
        ax4.set_yticklabels(labels)
        plt.colorbar(im, ax=ax4, label=get_label('队列长度', 'Queue Length'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "queue_analysis.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="q3_eval_decisions.csv", help="evaluation decisions csv path")
    parser.add_argument("--outdir", type=str, default="q3_plots", help="output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    ensure_dir(args.outdir)
    df = pd.read_csv(args.csv)

    print(f"加载数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"列名: {list(df.columns)}")

    # 基础图表
    if "reward" in df.columns:
        plot_reward(df, args.outdir)
        print("✓ 奖励分析图已生成")

    # 基站资源分配图
    bs_list = detect_base_stations(df)
    for bs in bs_list:
        plot_rb_per_bs(df, bs, args.outdir)
        plot_power_per_bs(df, bs, args.outdir)
    print(f"✓ 基站资源分配图已生成 ({len(bs_list)} 个基站)")

    # 新增分析图表
    plot_timeout_penalties(df, args.outdir)
    print("✓ 超时惩罚分析图已生成")
    
    plot_delay_analysis(df, args.outdir)
    print("✓ 时延变化分析图已生成")
    
    plot_markov_decision_analysis(df, args.outdir)
    print("✓ 马尔可夫决策分析图已生成")
    
    plot_queue_analysis(df, args.outdir)
    print("✓ 队列分析图已生成")

    print(f"所有图表已保存到 {args.outdir}")


if __name__ == "__main__":
    main()

