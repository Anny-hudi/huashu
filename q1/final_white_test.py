import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 完全重置所有样式
plt.rcdefaults()
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# 强制设置白色背景
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

# 设置seaborn样式
sns.set_style("white")

# 创建测试数据
x = np.random.randn(100)
y = np.random.randn(100)

# 创建图表
fig = plt.figure(figsize=(8, 6), facecolor='white')
ax = fig.add_subplot(111, facecolor='white')
ax.scatter(x, y, alpha=0.7, color='blue')
ax.set_title('最终白色背景测试', color='black')
ax.set_xlabel('X轴', color='black')
ax.set_ylabel('Y轴', color='black')
ax.grid(True, alpha=0.3)

# 确保所有元素都是白色背景
fig.patch.set_facecolor('white')
ax.patch.set_facecolor('white')

# 保存图片
plt.savefig('q1/final_white_test.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', transparent=False)
print("最终白色背景测试图片已保存到: q1/final_white_test.png")

# 创建JointGrid测试
fig2 = plt.figure(figsize=(8, 6), facecolor='white')
g = sns.JointGrid(x=x, y=y, height=6, ratio=3)
g.fig.patch.set_facecolor('white')
g.ax_joint.set_facecolor('white')
g.ax_marg_x.set_facecolor('white')
g.ax_marg_y.set_facecolor('white')

g.plot_joint(sns.scatterplot, alpha=0.7, color='red')
g.plot_marginals(sns.histplot, alpha=0.6, color='red')

# 保存JointGrid图片
g.fig.savefig('q1/final_jointgrid_test.png', dpi=300, bbox_inches='tight', 
              facecolor='white', edgecolor='none', transparent=False)
print("最终JointGrid测试图片已保存到: q1/final_jointgrid_test.png") 