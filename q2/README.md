# 问题2：网络切片资源分配解决方案

## 问题描述

在1000ms的仿真时间内，系统需要每100ms进行一次资源分配决策，总共10次决策。每次决策需要将50个资源块分配给三类切片（URLLC、eMBB、mMTC），使得整体用户服务质量达到最大。

## 解决方案

本项目提供了三种不同的解决方案：

### 1. 动态优化方法 (q2_dynamic_optimization.py)

- **算法**: 基于梯度下降的优化算法
- **特点**: 
  - 使用scipy.optimize.minimize进行优化
  - 考虑资源块倍数约束
  - 实时计算QoS并优化
- **优势**: 收敛速度快，结果稳定

### 2. 强化学习方法 (q2_reinforcement_learning.py)

- **算法**: Q-Learning
- **特点**:
  - 状态空间：用户任务到达概率
  - 动作空间：所有可能的资源分配组合
  - 奖励函数：基于QoS计算
- **优势**: 能够学习长期最优策略

### 3. 混合优化方法 (q2_hybrid_solution.py)

- **算法**: 结合动态优化和强化学习
- **特点**:
  - 同时运行两种方法
  - 选择QoS更高的结果
  - 提供方法使用统计
- **优势**: 结合两种方法的优点

## 核心算法

### 信号传输模型
```
SINR = (P_tx - φ - h) / (N_0 + I)
Rate = B * log2(1 + SINR)
```

### QoS计算
- **URLLC**: QoS = α^delay (延迟折扣)
- **eMBB**: QoS = min(1, rate/rate_SLA) (速率折扣)
- **mMTC**: QoS = allocated_RBs / total_RBs (接入比例)

### 资源约束
- 总资源块：50个
- URLLC：10的倍数
- eMBB：5的倍数
- mMTC：2的倍数

## 文件结构

```
q2/
├── q2_dynamic_optimization.py    # 动态优化解决方案
├── q2_reinforcement_learning.py  # 强化学习解决方案
├── q2_hybrid_solution.py         # 混合优化解决方案
├── q2_main.py                    # 主程序
└── README.md                     # 说明文档
```

## 运行方法

```bash
# 运行所有方法
python q2_main.py

# 单独运行动态优化
python q2_dynamic_optimization.py

# 单独运行强化学习
python q2_reinforcement_learning.py

# 单独运行混合优化
python q2_hybrid_solution.py
```

## 输出结果

每个方法都会输出：
- 每次决策的资源分配方案
- 各切片的QoS值
- 总体QoS统计信息
- 平均、最大、最小QoS

## 技术特点

1. **实时优化**: 根据当前任务队列和信道状态进行优化
2. **约束满足**: 严格满足资源块倍数约束
3. **QoS最大化**: 以用户服务质量为优化目标
4. **算法对比**: 提供多种算法实现和性能对比

## 依赖库

- numpy
- pandas
- matplotlib
- scipy
- random

## 数据文件

使用data_2目录下的数据文件：
- 用户任务流2.csv
- 用户位置2.csv
- 大规模衰减2.csv
- 小规模瑞丽衰减2.csv

# 混合资源分配算法可视化分析

## 新增绘图功能

本文件新增了6种专业的图表绘制功能，用于全面分析混合资源分配算法的性能：

### 1. 资源分配比例图 (Resource Allocation Chart)
- **类型**: 堆叠面积图
- **目的**: 展示三种切片在不同时间点的资源分配比例
- **输出**: `resource_allocation.png`

### 2. QoS性能对比图 (QoS Comparison Chart)
- **类型**: 分组柱状图
- **目的**: 比较三种切片在不同决策时刻的QoS表现
- **输出**: `qos_comparison.png`

### 3. 决策方法选择热力图 (Decision Method Heatmap)
- **类型**: 热力图
- **目的**: 展示混合决策中动态优化和强化学习的选择比例
- **输出**: `decision_heatmap.png`
- **依赖**: seaborn库

### 4. 任务队列动态图 (Queue Dynamics Chart)
- **类型**: 多线折线图
- **目的**: 展示任务队列长度的实时变化
- **输出**: `queue_dynamics.png`

### 5. 三维资源分配散点图 (3D Resource Space Chart)
- **类型**: 3D散点图
- **目的**: 展示资源分配的决策空间
- **输出**: `3d_resource_space.png`

### 6. 算法性能对比雷达图 (Performance Radar Chart)
- **类型**: 雷达图
- **目的**: 对比不同切片的平均QoS性能
- **输出**: `radar_chart.png`

## 使用方法

### 自动绘制所有图表
```python
# 运行仿真后自动绘制所有图表
hybrid_optimizer.plot_all_charts(results)
```

### 单独绘制特定图表
```python
# 绘制资源分配比例图
hybrid_optimizer.plot_resource_allocation(results)

# 绘制QoS性能对比图
hybrid_optimizer.plot_qos_comparison(results)

# 绘制决策方法选择热力图
hybrid_optimizer.plot_decision_heatmap(results)

# 绘制任务队列动态图
hybrid_optimizer.plot_queue_dynamics(results)

# 绘制三维资源分配散点图
hybrid_optimizer.plot_3d_resource_space(results)

# 绘制算法性能对比雷达图
hybrid_optimizer.plot_radar_chart(results)
```

## 依赖库

- numpy: 数值计算
- pandas: 数据处理
- matplotlib: 基础绘图
- scipy: 优化算法
- seaborn: 高级统计图表 (可选)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 注意事项

1. 所有图表都会自动保存到q2目录下
2. 图片分辨率设置为300 DPI，适合论文使用
3. 使用专业配色方案，确保图表美观
4. 支持中文字体显示
5. 如果seaborn库未安装，热力图功能将被跳过

## 图表特点

- **学术性**: 适合数学建模比赛和学术论文
- **专业性**: 使用标准的科研图表样式
- **美观性**: 采用现代设计理念和配色方案
- **实用性**: 提供全面的算法性能分析视角
