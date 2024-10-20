import pandas as pd
import numpy as np
df = pd.read_csv('clgdataset.csv')
df.head()
df.info()
df.isnull().sum()
X=df.iloc[: , 1:3].values
y=df.iloc[: , -1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=2)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
np.sqrt(X_train.shape[0])
k=9

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cm
