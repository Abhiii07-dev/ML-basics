"TP,FP,TN,FN"

from sklearn.metrics import confusion_matrix

y_True = [0,1,0,1,0,1,0,0,1]
y_pred = [1,1,0,1,0,0,0,1,1]


cm = confusion_matrix(y_True,y_pred)
print("Confusion matrix")
print(cm)