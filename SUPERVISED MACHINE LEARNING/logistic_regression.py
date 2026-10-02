"""
its diffrent from the linear regression because linear regression wlways deals with the numbers...
but logistic regression deals with categories and labels like (yes/no) , (pass/fail)
"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
x = [[1],[2],[3],[4],[5]]
y = [0,0,1,1,1]

model.fit(x,y)

hours = float(input("Enter the number of hours studies:"))

result = model.predict([[hours]])[0]      #[0] = list ke andar ki value milegi

if result == 0:
    print(f"based on your {hours} hours study, you're likely to fail!")

else :
    print(f"based on your {hours} hours study, you're likely to pass!")

