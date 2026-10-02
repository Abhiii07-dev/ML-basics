"""
4 steps:
1. choose value of K(how many neighbours to check)
2. find K
3. how many are in each class(yes or no)
4. predict the class with the majority vote
"""

from sklearn.neighbors import KNeighborsClassifier
x = [
    [180,7],
    [200,7.5],
    [250,8],           # wight and size of fruits
    [270,8.5],
    [333,9],
    [370,9.5]
]

y = [0,0,0,0,1,1]    # 0 = Apple & 1 = orange

model = KNeighborsClassifier(n_neighbors=3)    #CHECK 3 NEIGHBOURS
model.fit(x,y)

weight = float(input("Enter the wight of the fruit:"))
size = float(input("Enter the size of the fruit:"))

predction = model.predict([[weight,size]])[0]

if predction == 0:
    print(f"Based on size:{size} and weight:{weight} , the fruit may be Apple")

else:
    print(f"Based on size:{size} and weight:{weight} , the fruit may be Orange")

