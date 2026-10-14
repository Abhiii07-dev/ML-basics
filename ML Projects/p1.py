import pandas as  pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("smolstudents.csv")

x = data[["hours"]].values
y = data["scores"].values

model = LinearRegression()    
model.fit(x,y)

predicted_score = model.predict(x)

#evaluation

mae = mean_absolute_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse = np.sqrt(mse)

print(mae)
print(mse)
print(rmse)


# Train Test Split

x = data[["hours"]]
y = data[["scores"]]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)

print(f"Training Data : \n{x_train}")
print(f"Testing Data : \n{x_test}")
print(f"Training Target : \n{y_train}")
print(f"test Data : \n{y_test}")



# #extra prediction
# new_hour = float(input("Enter the hours study:"))
# new_pred = model.predict([[new_hour]])
# print(f"According to the {new_hour} hours studied , your score may be {new_pred}")


