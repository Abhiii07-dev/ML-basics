"""
Mean Absolute error
:
take the mistake difference
remove the negative sign
add
divine
:- 7.5

Mean Squared Error

mistakes square them 
add
divide total
:- 6.35

Root Mean Square Error
"""


from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np


real_scores = [90,60,80,100]
predicted_scores = [85,70,70,95]

mae = mean_absolute_error(real_scores,predicted_scores)
mse = mean_squared_error(real_scores,predicted_scores)
rmse = np.sqrt(mse)

print(f"MAE:{mae}")
print(f"MSE:{mse}")
print(f"RMSE:{rmse}")

