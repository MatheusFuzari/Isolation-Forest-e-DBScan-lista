import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
leucocitos =[2000,4000,5000,6500]
plaquetas = [100000,20000,80000,145000]
linfocitos = [2.3,4.5,6.5,4.4]
plt.scatter(leucocitos,plaquetas, c=linfocitos)
plt.xlabel("leucocitos")
plt.ylabel("plaquetas")
plt.colorbar(label='linfócitos')
plt.show()
exame_data = np.column_stack((leucocitos,plaquetas,linfocitos))
isolation_forest = IsolationForest(contamination=0.05, random_state=0)
isolation_forest.fit(exame_data[:,1:])
labels = isolation_forest.predict(exame_data[:, 1:])
n_anomalias = np.sum(labels == -1)
print("Anomalias: ", n_anomalias)

fig = plt.figure(figsize=(15,15))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(leucocitos,plaquetas,linfocitos, c=labels, cmap='coolwarm')
ax.set_xlabel("Leucócitos")
ax.set_ylabel("Plaquetas")
ax.set_zlabel("Linfócitos")
plt.show()