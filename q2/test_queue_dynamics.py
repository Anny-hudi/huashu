import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_queue_dynamics_test():
    """测试绘制任务队列动态图"""
    # 模拟队列长度数据
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 模拟不同切片的队列长度
    urllc_queue = [2, 3, 1, 4, 2, 3, 1, 2, 3, 1]
    embb_queue = [5, 4, 6, 3, 5, 4, 6, 5, 4, 6]
    mmtc_queue = [8, 7, 9, 6, 8, 7, 9, 8, 7, 9]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(times, urllc_queue, 'o-', label='URLLC', color='#FF6B6B', linewidth=2, markersize=6)
    plt.plot(times, embb_queue, 's-', label='eMBB', color='#4ECDC4', linewidth=2, markersize=6)
    plt.plot(times, mmtc_queue, 'd-', label='mMTC', color='#556270', linewidth=2, markersize=6)
    
    plt.xlabel('时间 (秒)', fontsize=12, fontweight='bold')
    plt.ylabel('队列长度', fontsize=12, fontweight='bold')
    plt.title('任务队列长度动态变化', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('q2/queue_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("队列动态图已生成并保存为 q2/queue_dynamics.png")

if __name__ == "__main__":
    plot_queue_dynamics_test() 