def learning_schedule(t):
    t0,t1 = 5, 50 # 학습 스케쥴 하이퍼파라미터
    return t0/(t+t1)

# 데이터 생성
import numpy as np
import matplotlib.pyplot as plt

X=2*np.random.rand(100,1)
y=4+3*X+np.random.randn(100,1)

# 정규방정식을 이용한 Theta값 구하기
X_b = np.c_[np.ones((100,1)) , X]
Theta = np.linalg.inv(X_b.transpose().dot(X_b)).dot(X_b.transpose().dot(y))
print("Theta",Theta)

#Theta를 이용한 선형회귀
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
y_pred = X_new_b.dot(Theta)
print("Linear Regression",y_pred)


#훈련을 위한 전처리
m = 100 #훈련 데이터 세트 개수

#Batch Gradient Descent
eta = 0.1 # learning rate
n_iterations = 1000
theta = np.random.randn(2,1)
for iter in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print("Batch Gradient Descent Method ", theta)

#Stochastic Gradient Descent
n_epochs = 50
theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m): # m은 훈련 데이터 세트 셈플의 수
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

print("Stochastic Gradient Descent Method",theta)

#matplotlib.pyplot을 이용해 출력
plt.plot(X_new,y_pred,"r-")
plt.plot(X,y,"b.")
plt.axis([0,2,0,15])
plt.show()