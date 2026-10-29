import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\ap910\Downloads\heart.csv")

print(df.head())
print(df.tail())
print(df.info)
print(df.describe())


print(df.isnull().sum())


# VISUALISATION

print(df.columns)


# plt.figure(figsize = (14,10))

# def plotting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var],kde = True)
#     plt.xlabel(var)
#     plt.tight_layout()


# plotting("Age",1)
# plotting("RestingBP",2)
# plotting("Cholesterol",3)
# plotting("MaxHR",4)

# plt.tight_layout(pad = 3.0)
# plt.show()



# sns.countplot(x = df["Sex"], hue=df["HeartDisease"])
# plt.show()

# sns.countplot(x = df["ChestPainType"],hue=df["HeartDisease"])
# plt.show()

# sns.countplot(x = df["RestingECG"], hue=df["HeartDisease"])
# plt.show()

# sns.countplot(x = df["ExerciseAngina"], hue=df["HeartDisease"])
# plt.show()

# sns.countplot(x = df["ST_Slope"], hue=df["HeartDisease"])
# plt.show()

# CLEANINNG

df.drop_duplicates(inplace=True)


# print(df['HeartDisease'].value_counts())
# print(df['RestingBP'].value_counts())


ch_mean = df.loc[df["Cholesterol"] != 0 , "Cholesterol"].mean()
df["Cholesterol"] = df["Cholesterol"].replace(0,ch_mean)
df["Cholesterol"] = df["Cholesterol"].round(2)




resting_bp_mean = df.loc[df["RestingBP"] != 0 , "Cholesterol"].mean()
df["RestingBP"] = df["RestingBP"].replace(0,resting_bp_mean)
df["RestingBP"] = df["RestingBP"].round(2)



# plt.figure(figsize = (14,10))

# def plotting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var],kde = True)
#     plt.xlabel(var)
#     plt.tight_layout()


# plotting("Age",1)
# plotting("RestingBP",2)
# plotting("Cholesterol",3)
# plotting("MaxHR",4)

# plt.tight_layout(pad = 3.0)
# plt.show()


# sns.boxplot(x = "HeartDisease" , y = "Cholesterol" , data = df)
# plt.show()

# sns.violinplot(x = "HeartDisease" , y = "Age" , data = df)
# plt.show()


# sns.heatmap(df.corr(numeric_only=True), annot = True)
# plt.show()



# PREPROCESSING

# print(df.head())

# print(df['RestingECG'].value_counts())
# print(df['ST_Slope'].value_counts())



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Sex"]= le.fit_transform(df["Sex"])
df["ExerciseAngina"]= le.fit_transform(df["ExerciseAngina"])


df = pd.get_dummies(df,columns=["ChestPainType","RestingECG","ST_Slope"],drop_first=True,dtype=int)

print(df.head())

# FEATURE ENGINEERING + MODEL EVALUATION

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

x = df.drop("HeartDisease",axis = 1)
y = df["HeartDisease"]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
x_train_sclaed = scaler.fit_transform(x_train)
x_test_sclaed = scaler.fit_transform(x_test)


models = {
    "logistic regression" : LogisticRegression(),
    "KNN" : KNeighborsClassifier(),
    "Naive Bias" : GaussianNB(),
    "SVM" : SVC(),
    "Decision Tree" : DecisionTreeClassifier()

}

result = []

for name,model in models.items():
    model.fit(x_train_sclaed,y_train)
    y_pred = model.predict(x_test_sclaed)
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    result.append({
        "model" : name,
        "accuracy" : round(acc,4),
        "f1 score" : round(f1,4)
    })
print(result)



# SAVING.............

import joblib

joblib.dump(models["logistic regression"],"Logistic_heart.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(x.columns.tolist(),"columns.pkl")