# ------------ 有节点背叛、不施加抗背叛机制的仿真 -----------
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import random
import os
from control import lqr

script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, 'figure')
os.makedirs(figure_dir, exist_ok=True)

# 字体: Arial, 字号 8–12 pt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['legend.fontsize'] = 8
# 线条粗细: 轴线 ~0.5 pt, 曲线 ~1.0 pt
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = 4
# 确保字体可嵌入 PDF/EPS（Type 42 = TrueType，避免 Type 3 位图字体）
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# 图片尺寸: ~14 cm 宽（1.5 栏）
matplotlib.rcParams['figure.figsize'] = (6.0, 4.5)

N = 50
TARGET_VALUE = 1
BETA = 0.8
a = 0.5

# 故障节点和被选择的节点
initial_chosen_nodes = [1, 9, 19, 32, 48]
chosen_nodes = initial_chosen_nodes.copy()
defection_nodes = [9, 19, 32]
replaced_nodes = []

# 初始随机生成强连通网络
G = nx.gnm_random_graph(N, N * N / 5)
while not nx.is_connected(G):
    rand1 = random.randint(0, N - 1)
    rand2 = random.randint(0, N - 1)
    if rand1 != rand2:
        G.add_edge(rand1, rand2)

"""
# 绘制图形
plt.figure(1)
pos = nx.spring_layout(G)  # 为图形设置布局
nx.draw(G, pos, node_size=700, edge_color='k', linewidths=1, font_size=15)
nx.draw_networkx_labels(G, pos)   # 在图中标注节点序号
"""

payoff_matrix_CAIPD = [4, -1, 5, 0]
# payoff_matrix_CAISD = [4.5, 4, 5, 0]

# 适应度
f = np.zeros(N)
delta_f = np.zeros((N, N))

# 初始随机生成对象的状态
x = np.random.rand(N)
# print(x)

# 网络的邻接矩阵表示
m = nx.to_numpy_array(G)
# print(m)

dx = np.zeros(N)

# 记录x每次迭代的值
x_record = [x.copy()]
energy_cost = []

# 边的权重
p = np.zeros((N, N))

# 网络中对象的度
degree = [G.degree(n) for n in G.nodes()]


def failure_recovery(faulty_nodes, chosen_nodes):
    avg_degree = [0 for i in range(N)]
    candidate_nodes = {}

    for i in range(N):
        if i in faulty_nodes and i in chosen_nodes:
            candidate_nodes[i] = []
            for j in range(N):
                if m[i, j] > 0:
                    avg_degree[i] += degree[j]
                if abs(degree[j] - degree[i]) < 5 and i != j:
                    candidate_nodes[i].append(j)
            avg_degree[i] /= degree[i]

    for i in range(N):
        if i in faulty_nodes and i in chosen_nodes:

            for j in candidate_nodes[i]:
                for k in range(N):
                    if m[j, k] > 0:
                        avg_degree[j] += degree[k]
                avg_degree[j] /= degree[j]
            min_difference = 100
            for j in candidate_nodes[i]:
                if abs(avg_degree[j] - avg_degree[i]) < min_difference:
                    min_difference = abs(avg_degree[j] - avg_degree[i])
                    candidate = j
            chosen_nodes.remove(i)
            chosen_nodes.append(candidate)
            replaced_nodes.append(candidate)
            print("节点", i, "故障，由节点", candidate, "替代")


def delta_fitness():
    global delta_f
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


def without_noise():
    delta_fitness()
    delta_ukk = np.zeros(N)
    for i in range(N):
        dx[i] = 0
        if i in chosen_nodes:
            for j in range(N):
                if m[i][j] > 0:
                    p[i][j] = BETA / (1 + math.exp(-abs(delta_f[j][i]))) / degree[j]
                    dx[i] += ukk(i, j)
            delta_ukk[i] = abs(dx[i] / degree[i])

        else:
            for j in range(N):
                if m[i][j] > 0:
                    p[i][j] = BETA / (1 + math.exp(-abs(delta_f[j][i]))) / degree[j]
                    dx[i] += p[i][j] * np.sign(x[j] - x[i]) * (abs(x[j] - x[i]) ** a)
        dx[i] /= degree[i]

    for i in range(N):
        if x[i] + dx[i] < 0:
            x[i] = 0
        elif x[i] + dx[i] > 1:
            x[i] = 1
        else:
            x[i] += dx[i]

    energy_cost.append(delta_ukk.copy())
    x_record.append(x.copy())


def ukk(node_i, node_j):
    delta = 0
    if node_i == node_j:
        for j in range(N):
            if m[node_i][j] > 0:
                delta += BETA / (1 + math.exp(-abs(delta_f[j][node_i]))) / degree[j]
    else:
        delta = -m[node_i][node_j] * (BETA / (1 + math.exp(-abs(delta_f[node_j][node_i]))) / degree[node_j])

    A = [[delta / degree[node_i]]]
    B = np.array([1])
    Q = np.array([0.01])
    R = np.array([1])
    k = lqr(A, B, Q, R)[0][0][0]
    return -k * (x[node_i] - TARGET_VALUE)


# 收敛判断函数
def are_elements_converged(array, target_value, tolerance=1e-3):
    """
    :param target_value: 目标值
    :param tolerance: 容忍度
    """
    return np.all(np.abs(array - target_value) < tolerance)


# 迭代运行
k = 0
while k < 20:
    delta_fitness()
    without_noise()
    k += 1

for node in defection_nodes:
    while node in chosen_nodes:
        #x[node] = 0.1
        chosen_nodes.remove(node)
#failure_recovery(defection_nodes, chosen_nodes)

while k < 450:
    delta_fitness()
    without_noise()
    k += 1

# 收敛结果展示
if are_elements_converged(x, TARGET_VALUE):
    print(f"在有限时间内收敛!")
else:
    print("无法在有限时间内收敛！")

# 绘制迭代收敛图像
plt.figure(2, dpi=300)
# 绘制其他节点
non_chosen_indices = [i for i in range(N) if i not in initial_chosen_nodes]
for idx in non_chosen_indices:
    node_record = [x_record[i][idx] for i in range(len(x_record))]
    plt.plot(node_record, linewidth=0.7)

# 绘制 chosen_nodes 的节点，用统一深红色
for node in initial_chosen_nodes:
    chosen_nodes_record = [x_record[i][node] for i in range(len(x_record))]
    plt.plot(chosen_nodes_record, linewidth=0.7, color='red', label='Chosen players' if node == initial_chosen_nodes[0] else "")

# 调整坐标轴显示
ax = plt.gca()
ax.tick_params(
    direction='in',  # 刻度线朝内
    length=4,  # 刻度线长度
    width=0.7,  # 刻度线粗细
    pad=7,  # 刻度标签与坐标轴的间距
    labelsize=10,  # 保持标签在外侧
    top=True,
    right=True
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

ax.set_ylim(0, y_max)
plt.axvline(x=20, linestyle='--', color='black', linewidth=0.6, dashes=(5, 2), ymin=0)
plt.xlabel("Iterations")
plt.ylabel("Player's strategy value ($x$)")
plt.title("BA-CAID induction simulation")
plt.legend()
plt.savefig(os.path.join(figure_dir, 'Fig_no_resistance_convergence.pdf'), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)

plt.figure(3, dpi=300)
for node in initial_chosen_nodes:
    chosen_nodes_record = [x_record[i][node] for i in range(len(x_record))]
    plt.plot(chosen_nodes_record, label=f'Player {node}', linewidth=0.8)

# 调整坐标轴显示
ax = plt.gca()
ax.tick_params(
    direction='in',  # 刻度线朝内
    length=4,  # 刻度线长度
    width=0.7,  # 刻度线粗细
    pad=7,  # 刻度标签与坐标轴的间距
    labelsize=10,  # 保持标签在外侧
    top=True,
    right=True
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

plt.axvline(x=20, linestyle='--', color='black', linewidth=0.6, dashes=(5, 2), ymin=0)
plt.xlabel("Iterations")
plt.ylabel("Player's strategy value ($x$)")
plt.title("Chosen players")
plt.legend()
plt.savefig(os.path.join(figure_dir, 'Fig_no_resistance_chosen_players.pdf'), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)



"""
plt.figure(5,dpi=300)
plt.plot(energy_cost,linewidth = 0.8)

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

plt.axvline(x=20, linestyle='--', color='black',linewidth=0.6, dashes=(5, 2), ymin=0)
plt.xlabel("iterations")
plt.ylabel("delta_ukk")
plt.title("Energy Cost")
plt.savefig(os.path.join(figure_dir, 'Fig_no_resistance_energy_cost.pdf'), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
"""

plt.show()
