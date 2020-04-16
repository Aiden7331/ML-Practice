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
discount_factor = 0.95 #미래에 대한 할인계수
possible_actions = [[0,1,2],[0,2],[1]] # MDP 과정. 자세한 내용은 마르코브 연쇄 참조

learning_rate0 = 0.5
learning_rate_decay = 0.1
n_iter=20000

s=0 #상태 0에서 시작

Q=np.full((3,3), -np.inf) # DP Algorithm

for state, actions in enumerate(possible_actions):
    Q[state][actions] = 0.0

for iter in range(n_iter):
    a=np.random.choice(possible_actions[s])
    sp = np.random.choice(range(3), p=T[s,a])
    reward = R[s,a,sp]
    learning_rate = learning_rate0/(1+ iter+learning_rate_decay)
    Q[s,a] = ((1-learning_rate)*Q[s,a]+learning_rate*(reward + discount_factor * np.max(Q[sp])))
    s=sp

print(Q)
print(np.argmax(Q, axis=1))