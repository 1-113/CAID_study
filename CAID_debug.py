import math
import numpy as np

N = 2
a=0.5
x=np.random.rand(N)

p = [[0.25, 2], [2, 0.25]]
dx = np.zeros(N)


for i in range(N):
    dx[i] = 0
    for j in range(N):
        dx[i] += p[i][j] * np.sign(x[j] - x[i]) * (abs(x[j] - x[i]) ** a)
    dx[i] = dx[i]

print(p[0][0] * np.sign(x[0] - x[0]) * (abs(x[0] - x[0]) ** a))
print(p[0][1] * np.sign(x[1] - x[0]) * (abs(x[1] - x[0]) ** a))
print(p[1][0] * np.sign(x[0] - x[1]) * (abs(x[0] - x[1]) ** a))
print(p[1][1] * np.sign(x[1] - x[1]) * (abs(x[1] - x[1]) ** a))
print(dx)
print(0.3**0.5)