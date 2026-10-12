"TP,FP,TN,FN"

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = cm.ravel()

y_true = [0,1,0,1,0,1,0,0,1]
y_pred = [1,1,0,1,0,0,0,1,1]

ConfusionMatrixDisplay.from_predictions(y_True, y_pred)

cm = confusion_matrix(y_True,y_pred)
print("Confusion matrix")
print(cm)

