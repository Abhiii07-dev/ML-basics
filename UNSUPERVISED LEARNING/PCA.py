import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data=  {
    "age" : [25,30,35,40,45],
    "income" : [40000,52000,60000,70000,80000],
    "spending" : [70,60,75,80,85],
    "savings" : [5000,6000,5500,8000,8500]
}
df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_function = scaler.fit_transform(df)

pca = PCA(n_components = 2)
pca_result = pca.fit_transform(scaled_function)

pca_df = pd.DataFrame(pca_result, columns = ["PCA1","PCA2"])

explained_variance = pca.explained_variance_ratio_
print("Varience captured by each PCA component:")
print(np.round(explained_variance*100,2))

plt.figure(figsize = (8,6))
plt.scatter(pca_df["PCA1"] , pca_df["PCA2"] , color = "black" ,s= 80)

plt.title("PCA projection (2D view)")
plt.xlabel("PCA1 maine pattern")
plt.ylabel("PCA2 minore pattern")
plt.grid(True)
plt.show()

print("New data with 2 features with PCA1 , PCA2")
print(pca_df)