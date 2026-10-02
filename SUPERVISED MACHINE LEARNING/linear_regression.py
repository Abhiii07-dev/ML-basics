"""
Linear Regressiion-

1.find a pattern in old data
2.make a straight line
3. predict value based on the line
y = m * x + b


its not about the line, its about the story
start with 5 rows(no need to start with 500 rows data)
accuracy is not always the goal
"""

from sklearn.linear_model import  LinearRegression

model = LinearRegression()

x = [[1],[2],[3],[4],[5]]
y = [40,50,65,75,90]

model.fit(x,y)                         # MODEL STARTS TRAINING FROM X AND Y DATA    
hours = float(input("no. of hours studied:"))

predicted_marks = model.predict([[hours]])               #2-D list because it shows that the model contains only one feature

print(f"Based on the gaiven data, the predicted marks is:\n{predicted_marks}")