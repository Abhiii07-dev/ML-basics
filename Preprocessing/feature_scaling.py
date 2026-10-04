
# import pandas as pd
# from sklearn.preprocessing import StandardScaler,MinMaxScaler



# "STANDARD SCALING" 
# # mean = 0 and std = 1

# scalar = StandardScaler()
# X_sclaed = scaler.fit_transform()

# scalar = MinMaxScaler()
# X_sclaed = scalar.fit_transform()














import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split                # SPLITS DATA INTO TEST AND TRANING 

data = {
    "study hours" : [1,2,3,4,5],
    "marks scored" : [40,50,60,70,80]

}
df = pd.DataFrame(data)

standard_scalar = StandardScaler()       #CREATES SCALAR OBJECT
standard_scaled = standard_scalar.fit_transform(df)   #LWARNS MEAN,STANDARD DEVIATION FROM EACH COLUMNS & THEM USE THIS INFO TO SCALE EACH VALUE

print(" Standard scalar value is:")
print(pd.DataFrame(standard_scaled,columns = ["study hours","marks scored"]))



minmax_scalar = MinMaxScaler()
minmax_scaled = minmax_scalar.fit_transform(df)     #LEARNS FROM THE MIN AND MAX VALUE OF THE DATA AND THEN SCALES IT FROM 0 TO 1

print("min max scalar value")
print(pd.DataFrame(minmax_scaled,columns = ["study hours","marks scored"])) 



x = df[["study hours"]]
y = df[["marks scored"]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# x_train,y_test = 80% training
# x_train,y_train = 20% testing
# random_state = 42 means that output will be same all the time


print("Training data")
print(x_train)

print("testing data")
print(x_test)


print("Training data")
print(y_train)

print("testing data")
print(y_test)



# THE VALUE SHOULD REVLOVE AROUND 0 , ITS BEST FOR THE MODEL TO TRAINING.....(for standard scaler) (for minmaxscaler its btw 0 to 1)