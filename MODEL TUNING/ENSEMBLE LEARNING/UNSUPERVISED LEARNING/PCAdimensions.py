import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


x,y = make_blobs(n_samples=500,n_features=5,centers=3,cluster_std=1.5,random_state=42)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
x_pca =     pca.fit_transform(x_scaled)

df_pca = pd.DataFrame(x_pca,columns=['PCA1','PCA2'])
df_pca['label'] = y

sns.scatterplot(data = df_pca,x ='PCA1' , y ='PCA2' , hue="label" , palette= "Set2")
plt.show()