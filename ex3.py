import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

exame_data = pd.DataFrame({
    "leucocitos" : [2000,4000,5000,6500],
    "plaquetas" : [100000,20000,80000,145000],
    "linfócitos" : [2.3,4.5,6.5,4.4]
}, index=[1,2,3,4])
plt.scatter(exame_data['leucocitos'],exame_data['plaquetas'], c=exame_data['linfócitos'])
plt.xlabel("leucocitos")
plt.ylabel("plaquetas")
plt.colorbar(label='linfócitos')
plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(exame_data[['leucocitos','plaquetas']])
dbscan = DBSCAN(eps=30, min_samples=2)
dbscan.fit(scaled_data)
labels = dbscan.labels_
exame_data['cluster'] = labels
mean_values = exame_data.groupby('cluster')['leucocitos'].mean()
n_clusters = len(set(labels))-(1 if -1 in labels else 0)

plt.scatter(exame_data['leucocitos'],exame_data['plaquetas'], c=exame_data['cluster'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(exame_data['leucocitos'],exame_data['plaquetas'],exame_data['linfócitos'], c=exame_data['cluster'])
plt.show()