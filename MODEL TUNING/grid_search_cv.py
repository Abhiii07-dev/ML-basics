import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("iris")
# print(df.head())

# print(df["species"].unique())

from sklearn.model_selection import train_test_split

x = df.drop("species",axis = 1)
y = df["species"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=
0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=5)  # for different n_neighbors , values should be different , but mine is same somehow

model_knn.fit(x_train,y_train)

# print(model_knn.score(x_test,y_test))


from sklearn.svm import SVC

# model_svm = SVC(C=30,kernel='rbf',gamma='auto')
# model_svm = SVC(C=10,kernel='linear',gamma='auto')  # chnaged this time
model_svm = SVC(gamma='auto')


model_svm.fit(x_train,y_train)

# print(model_svm.score(x_test,y_test))





# LETS USE GRID SEARCH CV

from sklearn.model_selection import GridSearchCV

classifier = GridSearchCV((model_svm),{
    "C" : [1,10,20,30],
    "kernel" : ["rbf","linear"]
    
},cv = 5 , return_train_score=False)

classifier.fit(x,y)

classifier.cv_results_

results = pd.DataFrame(classifier.cv_results_)
# print(results.columns)


# print(results[["param_C","param_kernel","mean_test_score"]])





'----------DO THE SAME DIFFERENT MODELS LIKE KNN....'




"RANDOM SEARCH CV"



from sklearn.model_selection import RandomizedSearchCV


classifier2 = RandomizedSearchCV((model_svm),{
    "C" : [1,10,20,30],
    "kernel" : ["rbf","linear"]
    
},n_iter = 4 ,cv = 5 , return_train_score=False)

classifier2.fit(x,y)

results2 = pd.DataFrame(classifier2.cv_results_)
# print(results)
print(results2[["param_C","param_kernel","mean_test_score"]])


#IT IS USED WEHN WE HAVE TO SELECT RANDOMLY A SMALLER CHUNK OF VALUES FROM VERT LAGRE ITERATIONS

