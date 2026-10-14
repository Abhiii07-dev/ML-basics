import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix



# df = pd.read_csv(r"C:\Users\ap910\Downloads\students performance.zip")

# print(f"Sample rows : \n{df.head()}")
# print(f"Dataset shape:")
# print(f"rows:{df.shape[0]}\n columns:{df.shape[1]}")
# print(f"Data info:\n{df.info()}")
# print(f"Summary Staistics:\n{df.describe(include = "all")}")
# print(f"Null values:\n{df.isnull().sum()}")







# IF THE VALUES IN AMT COLUMN ARE TEXT THAN WE HAVE TO USE LABELENCODER & GETDUMMIES
# HAVE TO FILL THE NULLL VALUES


df = pd.read_csv(r"C:\Users\ap910\Downloads\students performance.zip")
df_label = df.copy()


le = LabelEncoder()
df_label["internet_access_encoded"] = le.fit_transform(df["internet_access"])


# # FEATURE SCALING

features  = df[["study_hours","exam_score"]]
scaler = StandardScaler()
# df_scaled = df.copy()
standard_scaled = scaler.fit_transform(features)
print("Standard Sacled value is:")
print(pd.DataFrame(standard_scaled,columns=["study_hours","exam_score"]))

df_encoded = pd.get_dummies(df_label,columns=["course","sleep_quality","study_method","facility_rating"],dtype = int)
print(f"One hot data:\n{df_encoded}")


x = df[["study_hours"]]
y = df[["exam_score"]]

x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("Training data")
print(x_train)

print("testing data")
print(x_test)


print("Training data")
print(y_train)

print("testing data")
print(y_test)


# df_scaled[features] = scaler.fit_transform(df[features])

# df_encoded = pd.get_dummies(df_scaled,columns = ["gender","course","sleep_quality","study_method","facility_rating","exam_difficulty"])

# x = df_scaled[features]  # FEATURES
# y = df_scaled["exam_score"]



model = LinearRegression()
linear_x = df[["study_hours"]].values
linear_y = df["exam_score"].values

model.fit(linear_x,linear_y)

hours = float(input("Enter the no. of study hours:"))
predicted_score = model.predict([[hours]])

print(f"Based on the given data, the predicted marks is:\n{predicted_score}")



y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print(f" the mean absolute error is :{round(mae,3)}")
print(f" the mean squared value is :{round(mse,3)}")



# # print("Classification Report")
# # print(classification_report(y_test,y_pred))    # works on logicalregression
# # confusion = confusion_matrix(y_test,y_pred)

# # Visualtisation

# # plt.figure(figsize=(6,4))
# # sns.heatmap(confusion , annot="True", fmt="d", cmap= "Blues",xticklabels=["pass","fail"],yticklabels=["pass","fail"])

# # plt.xlabel("predicted")
# # plt.ylabel("Actual")
# # plt.title("Confusion matrix")
# # plt.tight_layout()
# # plt.show()

# # print("------------user input for prediction-------------")

# # try:
# #     studyhours = float(input("Enter the study hours:"))
# #     # studymethod = float(input("Enter the study method:"))
# #     facilityrating = float(input("Enter the facility rating:"))
# #     # sleepquality = float(input("Enter the exam diffilculty:"))

# #     user_input_df = pd.DataFrame([{
# #         studyhours : "study_hours",
# #         # studymethod : study_method,
# #         facilityrating : facility_rating,
# #         # sleepquality : sleep_quality

# #     }])

# #     user_input_scaled = scaler.transform(user_input_df)
# #     prediction = model.predict(user_input_scaled)
# #     result = "pass" if prediction == 1 else "fail" 
# #     print(f"prediction based on the user input : {result}")
# # except Exception as e:
# #     print("an error occured", e)
