import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import random

# 设置随机种子
random.seed(42)  # 设置Python内置随机种子
np.random.seed(42)  # 设置numpy随机种子

# 网络中对象数量
N = 50

a = 0.5
iteration_limit = 125

# 初始随机生成强连通网络
G = nx.gnm_random_graph(N, N * N / 5)
while not nx.is_connected(G):
    rand1 = random.randint(0, N - 1)
    rand2 = random.randint(0, N - 1)
    if rand1 != rand2:
        G.add_edge(rand1, rand2)

# 绘制图形
plt.figure(1)
pos = nx.spring_layout(G)  # 为图形设置布局
nx.draw(G, pos, node_size=700, edge_color='k', linewidths=1, font_size=15)

# 在图中标注节点序号
nx.draw_networkx_labels(G, pos)

payoff_matrix_CAIPD = [4, -1, 5, 0]
payoff_matrix_CAISD = [4.5, 4, 5, 0]

# 适应度
f = np.zeros(N)
delta_f = np.zeros((N, N))

# 初始随机生成对象的状态
x = np.random.rand(N)
# print(x)

# 网络的邻接矩阵表示
m = nx.to_numpy_array(G)
# print(m)

convergence_val = np.mean(x)
dx = np.zeros(N)

# 记录x每次迭代的值
x_record = [x.copy()]

# 边的权重
p = np.zeros((N, N))

# 网络中对象的度
degree = [G.degree(n) for n in G.nodes()]


# print(degree)

# CAIPD迭代函数
def caipd_iteration():
    # 适应度计算
    for i in range(N):
        f[i] = 0
        for j in range(N):
            f[i] += m[i][j] * ((payoff_matrix_CAIPD[0] - payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[2] +
                                payoff_matrix_CAIPD[3]) * x[i] * x[j] + (
                                       payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[3]) * x[j] + (
                                       payoff_matrix_CAIPD[2] - payoff_matrix_CAIPD[3]) * x[i] +
                               payoff_matrix_CAIPD[3])

    # 适应度差值
    for i in range(N):
        for j in range(N):
            delta_f[i][j] = f[i] - f[j]

    # 权重计算
    for i in range(N):
        for j in range(N):
            p[i][j] = m[i][j] * (0.5 / (1 + math.exp(-abs(delta_f[j][i]))) / degree[j])
    # print("p_matrix:", p)

    dx_total = 0
    for i in range(N):
        dx[i] = 0
        for j in range(N):
            dx[i] += p[i][j] * np.sign(x[j] - x[i]) * (abs(x[j] - x[i]) ** a)
        dx[i] = dx[i] / degree[i]
        dx_total += dx[i]
    # print(dx_total)

    for i in range(N):
        x[i] += dx[i]
    x_record.append(x.copy())  # 保存迭代结果


# CAISD迭代函数
def caisd_iteration():
    # 适应度计算
    for i in range(N):
        f[i] = 0
        for j in range(N):
            f[i] += m[i][j] * ((payoff_matrix_CAISD[0] - payoff_matrix_CAISD[1] - payoff_matrix_CAISD[2] +
                                payoff_matrix_CAISD[3]) * x[i] * x[j] + (
                                       payoff_matrix_CAISD[1] - payoff_matrix_CAISD[3]) * x[j] + (
                                       payoff_matrix_CAISD[2] - payoff_matrix_CAISD[3]) * x[i] +
                               payoff_matrix_CAISD[3])

    # 适应度差值
    for i in range(N):
        for j in range(N):
            delta_f[i][j] = f[i] - f[j]

    # 权重计算
    for i in range(N):
        for j in range(N):
            p[i][j] = m[i][j] * (0.5 / (1 + math.exp(-abs(delta_f[j][i]))) / degree[j])
    # print("p_matrix:", p)

    dx_total = 0
    for i in range(N):
        dx[i] = 0
        for j in range(N):
            dx[i] += p[i][j] * np.sign(x[j] - x[i]) * (abs(x[j] - x[i]) ** a)
        dx[i] = dx[i] / degree[i]
        dx_total += dx[i]
    # print(dx_total)

    for i in range(N):
        x[i] += dx[i]
    x_record.append(x.copy())  # 保存迭代结果


# 收敛判断函数
def are_elements_converged(array, target_value, tolerance=1e-3):
    """
    判断数组中的所有元素是否都收敛到了给定值的附近。

    :param array: 要检查的数组（numpy数组）
    :param target_value: 目标值
    :param tolerance: 容忍度
    :return: 如果所有元素都在目标值的容忍度范围内，返回True，否则返回False
    """
    return np.all(np.abs(array - target_value) < tolerance)


# 迭代运行
k = 0
while k < iteration_limit:
    caipd_iteration()
    k += 1

# 收敛结果展示
if are_elements_converged(x, convergence_val):
    print(f"在有限时间内收敛!")
else:
    print("无法在有限时间内收敛！")

# 绘制CAIPD迭代收敛图像
plt.figure(2, dpi=300)
plt.plot(x_record, linewidth=0.8)
plt.xlabel("Iterations")
plt.ylabel("Player's strategy value (x)")
# plt.text(0.01, 0.5, 'The strategy x of players', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
plt.title("BA-CAID convergence under the Prisoner's Dilemma")

# 调整坐标轴显示
ax = plt.gca()
ax.tick_params(
    direction='in',  # 刻度线朝内
    length=4,  # 刻度线长度
    width=0.7,  # 刻度线粗细
    pad=7,  # 刻度标签与坐标轴的间距
    labelsize=10, # 保持标签在外侧
    top = True,
    right = True
)

# 获取当前坐标轴的范围
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
ax.set_yticks(np.arange(0, np.ceil(y_max * 10) / 10, 0.1))

# 设置坐标轴位置和边界
ax.spines['left'].set_position('zero')
ax.spines['left'].set_bounds(0, y_max)

ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_bounds(0, x_max)

ax.spines['right'].set_position(('data', x_max))
ax.spines['right'].set_bounds(0, y_max)

ax.spines['top'].set_position(('data', y_max))
ax.spines['top'].set_bounds(0, x_max)

# 保存图像至指定路径
plt.savefig('/Volumes/SN770/研究生文件/CAID/figure/CAIPD_convergence.png', bbox_inches='tight')

x_record = [x_record[0]]
x = x_record[0].copy()

k = 0
while k < iteration_limit:
    caisd_iteration()
    k += 1

if are_elements_converged(x, convergence_val):
    print(f"在有限时间内收敛!")
else:
    print("无法在有限时间内收敛！")

# 绘制CAISD迭代收敛图像
plt.figure(3, dpi=300)
plt.plot(x_record, linewidth=0.8)
plt.xlabel("Iterations")
plt.ylabel("Player's strategy value (x)")
plt.title("BA-CAID convergence under the Snowdrift Dilemma")

# 调整坐标轴显示
ax = plt.gca()
ax.tick_params(
    direction='in',  # 刻度线朝内
    length=4,  # 刻度线长度
    width=0.7,  # 刻度线粗细
    pad=7,  # 刻度标签与坐标轴的间距
    labelsize=10, # 保持标签在外侧
    top = True,
    right = True
)

# 获取当前坐标轴的范围
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
ax.set_yticks(np.arange(0, np.ceil(y_max * 10) / 10, 0.1))

# 设置坐标轴位置和边界
ax.spines['left'].set_position('zero')
ax.spines['left'].set_bounds(0, y_max)

ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_bounds(0, x_max)

ax.spines['right'].set_position(('data', x_max))
ax.spines['right'].set_bounds(0, y_max)

ax.spines['top'].set_position(('data', y_max))
ax.spines['top'].set_bounds(0, x_max)

# 保存图像至指定路径
plt.savefig('/Volumes/SN770/研究生文件/CAID/figure/CAISD_convergence.png', bbox_inches='tight')

plt.show()
