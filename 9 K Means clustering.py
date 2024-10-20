from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset('iris')
df.head()
df['species'].unique()
df['species'].replace({'setosa':0,'versicolor':1,'virginica':2},inplace=True)
X = iris.data[:,2]
y = iris.data[:,3]
y.shape
plt.scatter(X,y)
X = iris.data[:,2:3]
X.shape
from sklearn.cluster import KMeans

inertias = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1,11),inertias,marker='o')
k=3
km = KMeans(n_clusters=k)
km.fit_predict(X)
plt.scatter(X,y,c=km.labels_)
