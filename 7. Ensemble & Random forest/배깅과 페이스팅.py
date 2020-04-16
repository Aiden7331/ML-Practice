from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

X_train, y_train = make_moons(n_samples=100,noise=None,random_state=None)
X_test, y_test = make_moons(n_samples=100,noise=None,random_state=None)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1) # 모든 CPU의 코어를 사용

bag_clf.fit(X_train,y_train)
y_pred = bag_clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))