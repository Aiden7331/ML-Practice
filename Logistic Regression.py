from sklearn import datasets
from sklearn.linear_model import LogisticRegression

import numpy as np

iris = datasets.load_iris()

list(iris.keys())

X = iris["data"][:, 3:] # 붗꽃 너비
Y = (iris["target"]== 2).astype(np.int)

log_reg= LogisticRegression()
log_reg.fit(X,Y)

X_new = np.linspace(0,3,1000).reshape(-1,1)
Y_proba = log_reg.predict_proba(X_new)

result=log_reg.predict([[1.7],[1]])
print(result[0])