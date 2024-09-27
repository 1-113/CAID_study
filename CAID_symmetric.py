import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import random

# 网络中对象数量
N = 5

a = 0.5

# 初始随机生成强连通网络
G = nx.gnm_random_graph(N, N - 1)
while not nx.is_connected(G):
    rand1 = random.randint(0, N-1)
    rand2 = random.randint(0, N-1)
    if rand1 != rand2:
        G.add_edge(rand1, rand2)

# 绘制图形
pos = nx.spring_layout(G)  # 为图形设置布局
nx.draw(G, pos, node_size=700, edge_color='k', linewidths=1, font_size=15)

# 在图中标注节点序号
nx.draw_networkx_labels(G, pos)

# 显示图形
plt.show()

payoff_matrix_CAIPD = [4, -1, 5, 0]
payoff_matrix_CAISD = [4.5, 4, 5, 0]

# 适应度
f = np.zeros(N)
delta_f = np.zeros((N, N))

# 初始随机生成对象的状态
x = np.random.rand(N)

# 网络的邻接矩阵表示
m = nx.to_numpy_array(G)
#print(m)

convergence_val = np.mean(x)
dx = np.zeros(N)

x_record = [x.copy()]

# 边的权重
p = np.zeros((N, N))

#网络中对象的度
degree =  [G.degree(n) for n in G.nodes()]
print(degree)



def iteration():
    for i in range(N):
        f[i] = 0
        for j in range(N):
            f[i] += m[i][j] * ((payoff_matrix_CAIPD[0] - payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[2] +
                                payoff_matrix_CAIPD[3]) * x[i] * x[j] + (
                                           payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[3]) * x[j] + (
                                           payoff_matrix_CAIPD[2] - payoff_matrix_CAIPD[3]) * x[i] +
                               payoff_matrix_CAIPD[3])

    for i in range(N):
        for j in range(N):
            delta_f[i][j] = f[i] - f[j]


    for i in range(N):
        for j in range(N):
            p[i][j] = delta_f[j][i]
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
    x_record.append(x.copy())


def are_elements_converged(array, target_value, tolerance=1e-3):
    """
    判断数组中的所有元素是否都收敛到了给定值的附近。

    :param array: 要检查的数组（numpy数组）
    :param target_value: 目标值
    :param tolerance: 容忍度
    :return: 如果所有元素都在目标值的容忍度范围内，返回True，否则返回False
    """
    return np.all(np.abs(array - target_value) < tolerance)


k = 0
while not are_elements_converged(x, convergence_val) and k < 1:
    iteration()
    k += 1
    print(delta_f)

if are_elements_converged(x, convergence_val):
    print(f"在有限时间内收敛，迭代次数为:{k}!")
else:
    print("无法在有限时间内收敛！")

x_iteration_matrix = np.array(x_record)

# fig = plt.figure(figsize=(8, 5))
for i in range(N):
    plt.plot(x_iteration_matrix[:, i])

plt.xlabel("iteration")
plt.ylabel("x value")
plt.title("CAID_test")
plt.show()
