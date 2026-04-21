import numpy as np
import math
import matplotlib.pyplot as plt

N = 50
a = 0.5
m = 5
k = 1

payoff_matrix_CAIPD = [m - k, -k, m, 0]
payoff_matrix_CAISD = [m - k / 2, m - k, m, 0]

f = np.zeros(N)
delta_f = np.zeros((N, N))
x = np.random.rand(N)
convergence_val = np.mean(x)
dx = np.zeros(N)
x_record = [x.copy()]
p = np.zeros((N, N))


# degree = np.zeros(N)


def iteration():
    for i in range(N):
        f[i] = 0
        for j in range(N):
            f[i] += (payoff_matrix_CAIPD[0] - payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[2] + payoff_matrix_CAIPD[
                3]) * x[i] * x[j] + (payoff_matrix_CAIPD[1] - payoff_matrix_CAIPD[3]) * x[j] + (
                                payoff_matrix_CAIPD[2] - payoff_matrix_CAIPD[3]) * x[i] + payoff_matrix_CAIPD[3]

    for i in range(N):
        for j in range(N):
            delta_f[i][j] = f[i] - f[j]

    for i in range(N):
        for j in range(N):
            p[i][j] = 0.5 / (1 + math.exp(-abs(delta_f[j][i])))
    # print("p_matrix:", p)

    dx_total = 0
    for i in range(N):
        dx[i] = 0
        for j in range(N):
            dx[i] += p[i][j] * np.sign(x[j] - x[i]) * (abs(x[j] - x[i]) ** a)
        dx[i] = dx[i] / N
        dx_total += dx[i]
    #print(dx_total)

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
while not are_elements_converged(x, convergence_val) and k <= 30:
    iteration()
    k += 1

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
