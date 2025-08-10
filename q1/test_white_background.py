import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 强制设置白色背景
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# 设置seaborn样式
sns.set_style("white")

# 创建测试数据
x = np.random.randn(100)
y = np.random.randn(100)

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
ax.scatter(x, y, alpha=0.7)
ax.set_title('测试白色背景')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 保存图片
plt.savefig('q1/test_white_background.png', dpi=300, bbox_inches='tight', facecolor='white')
print("测试图片已保存到: q1/test_white_background.png")

# 创建JointGrid测试
fig2 = plt.figure(figsize=(8, 6), facecolor='white')
g = sns.JointGrid(x=x, y=y, height=6, ratio=3)
g.fig.patch.set_facecolor('white')
g.plot_joint(sns.scatterplot, alpha=0.7)
g.plot_marginals(sns.histplot, alpha=0.6)
g.ax_joint.set_facecolor('white')
g.ax_marg_x.set_facecolor('white')
g.ax_marg_y.set_facecolor('white')

# 保存JointGrid图片
g.fig.savefig('q1/test_jointgrid_white.png', dpi=300, bbox_inches='tight', facecolor='white')
print("JointGrid测试图片已保存到: q1/test_jointgrid_white.png") 