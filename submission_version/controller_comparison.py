# ------------- 和使用其他控制器进行诱导的对比：收敛情况及能量消耗 ----------------
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
from control import lqr
import os
import matplotlib

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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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
G = nx.gnm_random_graph(N, N * N / 5, seed=SEED)
while not nx.is_connected(G):
    rand1 = random.randint(0, N - 1)
    rand2 = random.randint(0, N - 1)
    if rand1 != rand2:
        G.add_edge(rand1, rand2)

payoff_matrix_CAIPD = [4, -1, 5, 0]

class OriginalController:
    def __init__(self):
        self.f = np.zeros(N)
        self.delta_f = np.zeros((N, N))
        self.x = np.random.rand(N)
        self.x_record = [self.x.copy()]
        self.energy_cost = []
        self.p = np.zeros((N, N))
        self.dx = np.zeros(N)
        self.chosen_nodes = initial_chosen_nodes.copy()
        self.degree = [G.degree(n) for n in G.nodes()]
        self.m = nx.to_numpy_array(G)
        self.convergence_iteration = -1
        self.total_energy = 0

    def delta_fitness(self):
        for i in range(N):
            self.f[i] = 0
            for j in range(N):
                self.f[i] += self.m[i][j] * ((payoff_matrix_CAIPD[0] - payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[2] +
                                              payoff_matrix_CAIPD[3]) * self.x[i] * self.x[j] + (
                                                     payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[3]) * self.x[j] + (
                                                     payoff_matrix_CAIPD[2] - payoff_matrix_CAIPD[3]) * self.x[i] +
                                             payoff_matrix_CAIPD[3])
        for i in range(N):
            for j in range(N):
                self.delta_f[i][j] = self.f[i] - self.f[j]

    def ukk(self, node_i, node_j):
        delta = 0
        if node_i == node_j:
            for j in range(N):
                if self.m[node_i][j] > 0:
                    delta += BETA / (1 + math.exp(-abs(self.delta_f[j][node_i]))) / self.degree[j]
        else:
            delta = -self.m[node_i][node_j] * (
                        BETA / (1 + math.exp(-abs(self.delta_f[node_j][node_i]))) / self.degree[node_j])

        A = [[delta / self.degree[node_i]]]
        B = np.array([1])
        Q = np.array([0.01])
        R = np.array([1])
        k = lqr(A, B, Q, R)[0][0][0]
        return -k * (self.x[node_i] - TARGET_VALUE)

    def step(self):
        self.delta_fitness()
        delta_ukk = np.zeros(N)
        for i in range(N):
            self.dx[i] = 0
            if i in self.chosen_nodes:
                for j in range(N):
                    if self.m[i][j] > 0:
                        self.p[i][j] = BETA / (1 + math.exp(-abs(self.delta_f[j][i]))) / self.degree[j]
                        self.dx[i] += self.ukk(i, j)
                delta_ukk[i] = abs(self.dx[i] / self.degree[i])
            else:
                for j in range(N):
                    if self.m[i][j] > 0:
                        self.p[i][j] = BETA / (1 + math.exp(-abs(self.delta_f[j][i]))) / self.degree[j]
                        self.dx[i] += self.p[i][j] * np.sign(self.x[j] - self.x[i]) * (abs(self.x[j] - self.x[i]) ** a)
            self.dx[i] /= self.degree[i]

        for i in range(N):
            if self.x[i] + self.dx[i] < 0:
                self.x[i] = 0
            elif self.x[i] + self.dx[i] > 1:
                self.x[i] = 1
            else:
                self.x[i] += self.dx[i]

        self.energy_cost.append(delta_ukk.copy())
        self.x_record.append(self.x.copy())
        self.total_energy += np.sum(delta_ukk)

    def run(self, max_iterations=450):
        for k in range(max_iterations):
            self.step()
            if self.convergence_iteration == -1 and self.are_elements_converged(self.x, TARGET_VALUE):
                self.convergence_iteration = k + 1
        if self.convergence_iteration == -1:
            self.convergence_iteration = max_iterations

    def are_elements_converged(self, array, target_value, tolerance=1e-3):
        return np.all(np.abs(array - target_value) < tolerance)


# ==================== 传统PID控制器 ====================
class PIDController:
    def __init__(self):
        self.f = np.zeros(N)
        self.delta_f = np.zeros((N, N))
        self.x = np.random.rand(N)
        self.x_record = [self.x.copy()]
        self.energy_cost = []
        self.p = np.zeros((N, N))
        self.dx = np.zeros(N)
        self.chosen_nodes = initial_chosen_nodes.copy()
        self.degree = [G.degree(n) for n in G.nodes()]
        self.m = nx.to_numpy_array(G)
        self.convergence_iteration = -1
        self.total_energy = 0

        # PID参数
        self.Kp = 0.01  # 比例增益
        self.Ki = 0.01  # 积分增益
        self.Kd = 0.00  # 微分增益

        # 积分和微分项存储
        self.integral_error = {node: 0 for node in self.chosen_nodes}
        self.prev_error = {node: 0 for node in self.chosen_nodes}

    def delta_fitness(self):
        for i in range(N):
            self.f[i] = 0
            for j in range(N):
                self.f[i] += self.m[i][j] * ((payoff_matrix_CAIPD[0] - payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[2] +
                                              payoff_matrix_CAIPD[3]) * self.x[i] * self.x[j] + (
                                                     payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[3]) * self.x[j] + (
                                                     payoff_matrix_CAIPD[2] - payoff_matrix_CAIPD[3]) * self.x[i] +
                                             payoff_matrix_CAIPD[3])
        for i in range(N):
            for j in range(N):
                self.delta_f[i][j] = self.f[i] - self.f[j]

    def pid_control(self, node_i):
        """PID控制器计算控制输入"""
        error = TARGET_VALUE - self.x[node_i]

        # 积分项
        self.integral_error[node_i] += error
        self.integral_error[node_i] = np.clip(self.integral_error[node_i], -10, 10)  # 防止积分饱和

        # 微分项
        derivative = error - self.prev_error[node_i]
        self.prev_error[node_i] = error

        # PID输出
        control = self.Kp * error + self.Ki * self.integral_error[node_i] + self.Kd * derivative

        # 限制控制量范围
        control = np.clip(control, -0.5, 0.5)

        return control

    def step(self):
        self.delta_fitness()
        delta_ukk = np.zeros(N)

        for i in range(N):
            self.dx[i] = 0
            if i in self.chosen_nodes:
                # 使用PID控制器
                for j in range(N):
                    if self.m[i][j] > 0:
                        self.p[i][j] = BETA / (1 + math.exp(-abs(self.delta_f[j][i]))) / self.degree[j]
                        self.dx[i] += self.pid_control(i)
                delta_ukk[i] = abs(self.dx[i] / self.degree[i])
            else:
                for j in range(N):
                    if self.m[i][j] > 0:
                        self.p[i][j] = BETA / (1 + math.exp(-abs(self.delta_f[j][i]))) / self.degree[j]
                        self.dx[i] += self.p[i][j] * np.sign(self.x[j] - self.x[i]) * (abs(self.x[j] - self.x[i]) ** a)
            self.dx[i] /= self.degree[i]

        for i in range(N):
            if self.x[i] + self.dx[i] < 0:
                self.x[i] = 0
            elif self.x[i] + self.dx[i] > 1:
                self.x[i] = 1
            else:
                self.x[i] += self.dx[i]

        self.energy_cost.append(delta_ukk.copy())
        self.x_record.append(self.x.copy())
        self.total_energy += np.sum(delta_ukk)

    def run(self, max_iterations=450):
        for k in range(max_iterations):
            self.step()
            if self.convergence_iteration == -1 and self.are_elements_converged(self.x, TARGET_VALUE):
                self.convergence_iteration = k + 1
        if self.convergence_iteration == -1:
            self.convergence_iteration = max_iterations

    def are_elements_converged(self, array, target_value, tolerance=1e-3):
        return np.all(np.abs(array - target_value) < tolerance)

# 记录运行时间
start_time = time.time()

# 运行原控制器
print("\n[1] 运行原LQR控制器...")
original_ctrl = OriginalController()
original_ctrl.run(max_iterations=450)
original_time = time.time() - start_time
print(f"    运行时间: {original_time:.4f}秒")
print(f"    收敛迭代次数: {original_ctrl.convergence_iteration}")
print(f"    总能量消耗: {original_ctrl.total_energy:.6f}")

random.seed(SEED)
np.random.seed(SEED)

# 运行PID控制器
start_time = time.time()
print("\n[2] 运行传统PID控制器...")
pid_ctrl = PIDController()
pid_ctrl.run(max_iterations=450)
pid_time = time.time() - start_time
print(f"    运行时间: {pid_time:.4f}秒")
print(f"    收敛迭代次数: {pid_ctrl.convergence_iteration}")
print(f"    总能量消耗: {pid_ctrl.total_energy:.6f}")



# # 图1: 本文提出的最优诱导策略收敛曲线
# fig1 = plt.figure(dpi=300)
# non_chosen_indices = [i for i in range(N) if i not in original_ctrl.chosen_nodes]
# ax1 = plt.gca()
# for idx in non_chosen_indices:
#     node_record = [original_ctrl.x_record[i][idx] for i in range(len(original_ctrl.x_record))]
#     ax1.plot(node_record, linewidth=0.7)
# for node in original_ctrl.chosen_nodes:
#     chosen_nodes_record = [original_ctrl.x_record[i][node] for i in range(len(original_ctrl.x_record))]
#     ax1.plot(chosen_nodes_record, linewidth=0.7, color='orange', label='Chosen players' if node == original_ctrl.chosen_nodes[0] else "")
# ax1.set_title("Proposed Optimal Induction")
# ax1.set_xlabel("Iterations")
# ax1.set_ylabel("Player's strategy value ($x$)")
# ax1.tick_params(direction='in', length=4, width=0.7, pad=7, labelsize=10, top=True, right=True)
# ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
# ax1.set_xlim(left=0)
# ax1.set_ylim(bottom=0)
# ax1.grid(False)
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, 'Fig_comparison_lqr_convergence.pdf'), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
# print("    已保存: comparison_convergence_lqr.png")

# 图2: 传统PID控制器收敛曲线
fig2 = plt.figure(dpi=300)
non_chosen_indices_pid = [i for i in range(N) if i not in pid_ctrl.chosen_nodes]
ax2 = plt.gca()
for idx in non_chosen_indices_pid:
    node_record = [pid_ctrl.x_record[i][idx] for i in range(len(pid_ctrl.x_record))]
    ax2.plot(node_record, linewidth=0.7)
for node in pid_ctrl.chosen_nodes:
    chosen_nodes_record = [pid_ctrl.x_record[i][node] for i in range(len(pid_ctrl.x_record))]
    ax2.plot(chosen_nodes_record, linewidth=0.7, color='blue',
             label='Chosen players' if node == pid_ctrl.chosen_nodes[0] else "")
ax2.set_title("Traditional PID Controller")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Player's strategy value ($x$)")
ax2.tick_params(direction='in', length=4, width=0.7, pad=7, labelsize=10, top=True, right=True)
ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'Fig_comparison_pid_convergence.pdf'), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
print("    已保存: comparison_convergence_pid.png")

# 图3: 资源消耗对比（每步能量消耗）
fig3 = plt.figure(dpi=300)

original_energy_per_step = [np.sum(cost) for cost in original_ctrl.energy_cost]
pid_energy_per_step = [np.sum(cost) for cost in pid_ctrl.energy_cost]

ax = plt.gca()
ax.plot(original_energy_per_step, linewidth=1.0, color='green', label='Proposed Optimal Induction')
ax.plot(pid_energy_per_step, linewidth=1.0, color='blue', label='PID Controller')
ax.set_title("Energy Consumption")
ax.set_xlabel("Iterations")
ax.set_ylabel("Energy Consumption per Step")
ax.tick_params(direction='in', length=4, width=0.7, pad=7, labelsize=10, top=True, right=True)
ax.legend(loc='upper right')
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.5)
ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'Fig_comparison_energy.pdf'), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
print("    已保存: comparison_energy_per_step.png")

plt.show()
