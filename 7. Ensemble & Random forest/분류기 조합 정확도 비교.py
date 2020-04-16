from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons

X_train, y_train = make_moons(n_samples=100, noise=0.15, random_state=42)
X_test, y_test = make_moons(n_samples= 100, noise=None,random_state=None)
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC();

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf',rnd_clf),('svc',svm_clf)],
    voting='hard')

from sklearn.metrics import accuracy_score
for clf in(log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

