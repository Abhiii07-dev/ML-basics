"""
accuracy = "correct_prediction/total_prediction"
precision= out of everything I predicted positive, how many really were
recall = out of everything actually positive, how many did I catch
F1 score = balancing precion and recall data (harmonic mean of precision and recall)
"""
#it's not used to train or  predict thr data, its only to scoring the results

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report      


#True answers (What actually happened)
y_True = [1,0,1,0,1,0]

#Model's Prediction
y_pred = [1,0,0,0,1,0]

#evaluation
print(f"accuracy:{accuracy_score(y_True,y_pred)}")
print(f"precision:{precision_score(y_True,y_pred)}")
print(f"recall:{recall_score(y_True,y_pred)}")
print(f"F1 score:{f1_score(y_True,y_pred)}")



x = classification_report(y_True, y_pred)
print(x)