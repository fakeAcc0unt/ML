from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y= iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)
X_test.shape
X_train.shape
from sklearn.tree import DecisionTreeClassifier
clf =  DecisionTreeClassifier(criterion='gini',max_depth=5,splitter='random')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score ,mean_squared_error
accuracy_score(y_test,y_pred)

mean_squared_error(y_test,y_pred)
from sklearn.tree import plot_tree
plot_tree(clf)
