import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Customers" : ["raju","billa","dogesh","MC2","binota"],
    "age" : [17,39,27,22,19],
    "spending" :[100,500,250,475,300]
}
df = pd.DataFrame(data)

x = df[["age","spending"]]

model = KMeans(n_clusters=2, random_state = 42, n_init = 10)

# random_state = 42(stable data)
# n_clusters = 2 ( 2 groups)
# n_init = 10 time clustering and then will t=give the best ouput from those

df["group"] = model.fit_predict(x)


plt.figure(figsize=(6,5))
for group in df["group"].unique():
    group_data = df[df["group"]==group]
    plt.scatter(group_data["age"] , group_data["spending"],label = f"Group {group}")

plt.xlabel("Age")
plt.ylabel("Spending")
plt.title("Costumer sgement (K-means)")
plt.legend()
plt.grid(True)
plt.show()


print(df)