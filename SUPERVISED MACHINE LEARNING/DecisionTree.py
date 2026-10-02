from sklearn.tree import DecisionTreeClassifier

x = [
    [7,2],   
    [8,4],      #size and colour of fruit
    [9,8],
    [10,9]
]
y = [0,0,1,1]

model = DecisionTreeClassifier()
model.fit(x,y)

size = float(input("Enter the size of the fruit:"))
shade = float(input("Enter the shade of the fruit:"))

result = model.predict([[size,shade]])

if result == 0:
    print(f" This is laikely to be an apple!")
else:
    print(f" This is likely to be an orange!")
