import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

Data = sns.load_dataset("titanic")
df = pd.DataFrame(Data)
# print(df)

df_label = df.copy() 
le = LabelEncoder()

df_label["sex_encoder"] = le.fit_transform(df_label["sex"])      # WILL LEARN THE DATA AND THEN CONVERT IN INTO 1 AND 0
df_label["alone_encoder"] = le.fit_transform(df_label["alone"])

print(f"Label encoded data:\n{df_label[["sex","sex_encoder","alone","alone_encoder"]]}")


df_encoded = pd.get_dummies(df_label,columns=["embark_town"])  #convert text into smol birary(get_dummy)
print("\nOne hot encoded Data(embark_town)")                     #  FOR MULTIPLE 
print(df_encoded)

