import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
corrente_eletrica = np.array([5, 10, 14, 2, 1.5, 6])
tempo = np.array([1,2,4,6,7,10])

dados_manutencao = np.column_stack((corrente_eletrica,tempo))
isolation_forest = IsolationForest(contamination=0.05, random_state=0)
isolation_forest.fit(dados_manutencao[:,1:])
labels = isolation_forest.predict(dados_manutencao[:, 1:])
n_anomalias = np.sum(labels == -1)
print("Anomalias: ", n_anomalias)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(corrente_eletrica, tempo,c=labels, cmap='coolwarm')
ax.set_xlabel("Corrente elétrica")
ax.set_ylabel("Tempo")
plt.show()