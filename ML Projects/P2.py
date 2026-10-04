import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data = pd.read_csv(r"C:\Users\ap910\Downloads\s1.zip")

x = data[["study_hours_per_day"]]
y = data["exam_score"]

model = LinearRegression()
model.fit(x,y)
predicted_score = model.predict(x)

mae = mean_absolute_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse = np.sqrt(mse)
r2 = r2_score(y,predicted_score)

print(f" the mean absolute erroe is :{round(mae,3)}")
print(f" the mean squared value is :{round(mse,3)}")
print(f" the root mean squared value is :{round(rmse,3)}")
print(f"r2 score ( Model accuracy) :{round(r2,4)}")



#HISTOGRAM


plt.figure(figsize=(10,6))
plt.hist(data["exam_score"],bins = 70 ,color="maroon" , edgecolor = "black")
plt.title("Scores of students according to hours  studied")
plt.xlabel("Final exams score")
plt.ylabel("Hours studied")
plt.grid(True)
plt.show()


# SCATTER PLOT

plt.figure(figsize=(10,6))
plt.scatter(x,y ,color="green" , label = "Actual scores")
plt.plot(x,predicted_score,color = "pink" , label = " Predicted Scores (Regression Lines )")
plt.title("Scores of students according to hours  studied")
plt.xlabel("study hours per day")
plt.ylabel("Final outuput")
plt.grid(True)
plt.show()



new_hours = float(input("Enter the hours studies:"))
predicted_score = model.predict([[new_hours]])
print(f"Predicted final score according to {new_hours} hours study is : {predicted_score}")