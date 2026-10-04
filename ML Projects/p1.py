import pandas as  pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
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

print(f" the mean absolute erroe is :{mae}")
print(f" the mean squared value is :{mse}")
print(f" the root mean squared value is :{rmse}")


#extra prediction
new_hour = float(input("Enter the hours study:"))
new_pred = model.predict([[new_hour]])
print(f"According to the {new_hour} hours studied , your score may be {new_pred}")


