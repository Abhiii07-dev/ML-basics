import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



x,y_true = make_blobs(n_samples=100,centers=3,cluster_std=0.60,random_state=42)

df = pd.DataFrame(x,columns = ['Feature1','Feature2'])
# print(df)

scaler = StandardScaler()

x_scaled = scaler.fit_transform(df)

inertia = []
k_range = range(1,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

# print(inertia)


# plt.plot(k_range,inertia,marker = 'o')
# plt.show()   # ELBOW SHAPED..................

kmeans_final = KMeans(n_clusters=3,random_state=42)
cluster_labels = kmeans_final.fit_predict(x_scaled)
# print(cluster_labels)

df['cluster'] = cluster_labels



sns.scatterplot(x = df['Feature1'],
                y = df['Feature2'],
                hue= df['cluster'],
                palette='viridis'
)


plt.show()





'DISADVATNAGES OF KMEANS CLUSTER'


# 1. INITIALISE THE K VALUES
# 2. OUTLIERS WILL MOVE TO THE CENTRE PONIT
# 3. NON CIRCULAR DTATA