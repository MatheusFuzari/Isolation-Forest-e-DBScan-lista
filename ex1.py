import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

imoveis_data = pd.DataFrame({
    "area":[120,145,80,160,200,90,110,130,180,160],
    "valor":[300,450,550,600,350,420,550,780,350,575],
    "dist_praia":[15,15,8,25,12,15,22,8,5,14]
}, index=[1,2,3,4,5,6,7,8,9,10])
plt.scatter(imoveis_data['area'],imoveis_data['dist_praia'], c=imoveis_data['valor'])
plt.xlabel("Área")
plt.ylabel("Distância praia")
plt.colorbar(label='Valor')
plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(imoveis_data[['area','dist_praia']])
dbscan = DBSCAN(eps=30, min_samples=2)
dbscan.fit(scaled_data)
labels = dbscan.labels_
imoveis_data['cluster'] = labels
mean_values = imoveis_data.groupby('cluster')['valor'].mean()
n_clusters = len(set(labels))-(1 if -1 in labels else 0)

plt.scatter(imoveis_data['area'],imoveis_data['dist_praia'], c=imoveis_data['cluster'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(imoveis_data['area'],imoveis_data['valor'],imoveis_data['dist_praia'], c=imoveis_data['cluster'])
plt.show()