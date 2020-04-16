import numpy as np

nan = np.nan
T = np.array([
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
    [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
])
R = np.array([
    [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
    [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]],
])

possible_actions = [[0,1,2],[0,2],[1]] # MDP 과정. 자세한 내용은 마르코브 연쇄 참조

Q = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0

discount_factor = 0.95
n_iterations = 100

for iteration in range(n_iterations):
    Q_prev = Q.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q[s,a] = np.sum([
                T[s,a,sp] * (R[s,a,sp] + discount_factor*np.max(Q_prev[sp]))
                for sp in range(3)
            ])

print(Q)
print(np.argmax(Q, 1))
