from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os

PROJECT_ROOT_DIR = "."
CHAPTER_ID = ""

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)


iris =load_iris()
X = iris.data[:,2:] #꽃잎의 길이와 너비
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)

# class predict
print(tree_clf.predict_proba([[5,1.5]]))
result = tree_clf.predict([[5,1.5]])
if(result[0]==1):
    print("Iris-Virginica")


# create tree graph
export_graphviz(
    tree_clf,
    out_file=os.path.join("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

