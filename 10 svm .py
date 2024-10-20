from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y= iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)
from sklearn.svm import SVC

clf=SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

import matplotlib.pyplot as plt
X_train_reduced = X_train[:,:2]
clf.fit(X_train_reduced,y_train)
plt.plot_decision_boundary(X_test[:,:2],y_test,clf)
