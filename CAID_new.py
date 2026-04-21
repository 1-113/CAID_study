import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import random

BETA = 0.5


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# 网络中对象数量
N = 50

b =
c =

iteration_limit = 200

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
fitness = np.zeros(N)
delta_f = np.zeros((N, N))

# 初始随机生成对象的状态
x = np.random.rand(N)
# print(x)

# 网络的邻接矩阵表示
w = nx.to_numpy_array(G)
# print(m)

convergence_val = np.mean(x)
dx = np.zeros(N)

# 记录x每次迭代的值
x_record = [x.copy()]

# 边的权重
p = np.zeros((N, N))

h = np.zeros((N, N))
# 网络中对象的度
degree = [G.degree(n) for n in G.nodes()]


def iteration():
    for i in range(N):
        temp = 0
        for j in range(N):
            temp += w[i][j] * x[j]
        fitness[i] = -degree[i] * c * x[i] + b * temp

    for i in range(N):
        for j in range(N):
            p[i][j] = w[i][j] * sigmoid(BETA * (fitness[j] - fitness[i]))

    for i in range(N):
        temp = 0
        for j in range(N):
            temp += p[i][j] * (x[j] - x[i])
        h[i] = temp / degree[i]


def multi_node_control():


def failure_recovery():
    for i in range(N):
        if i #outofcontrol
            avg_cooperation[i] += w[i][j] * x[i]

        for j in range(N):
            if delta_degree < 3:
                avg_cooperation[j]+=w[i][j]*x[j]

plt.show()
