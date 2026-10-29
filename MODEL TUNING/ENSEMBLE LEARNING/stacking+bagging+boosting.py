from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")



df = sns.load_dataset('iris')

x= df.drop('species',axis = 1)
y = df["species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42,stratify=y_encoded)

base_leanrners = [
    ('dt',DecisionTreeClassifier(random_state=42)),
    ('svc',SVC(probability=True,kernel='rbf',random_state=42)),
    ('lr',LogisticRegression(max_iter=1000))

]

meta_learner = LogisticRegression(max_iter=1000)

stacking_clf = StackingClassifier(
    estimators=base_leanrners,
    final_estimator=meta_learner,
    cv = 5
)

stacking_clf.fit(x_train,y_train)

# y_pred = stacking_clf.predict(x_test)

# accuracy = accuracy_score(y_test,y_pred)
# print(accuracy)



'-------------------------------------BAGGING---------------------------------------------'


#  FOR OVERFITTED MODELS


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,# nuber of tress
    max_depth=None , # Let trees grow fully
    random_state=42
)

rf_model.fit(x_train,y_train)


# y_pred = rf_model.predict(x_test)


# accuracy = accuracy_score(y_test,y_pred)

# print(accuracy)




'-----------------------------------------------BOOSTING--------------------------------------------------------------------'


#  FOR UNDERFITTED MODELS

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier

from xgboost import XGBClassifier

ada_model = AdaBoostClassifier(n_estimators=100,random_state=42)
ada_model.fit(x_train,y_train)

# y_pred = ada_model.predict(x_test)

# accuracy = accuracy_score(y_test,y_pred)

# print(accuracy)

gb_model=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,random_state=42)
gb_model.fit(x_train,y_train)

# y_pred = gb_model.predict(x_test)

# accuracy = accuracy_score(y_test,y_pred)

# print(accuracy)


xgb_model= XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,use_label_encoder = False,eval_metric = 'mlogloss',random_state=42)

xgb_model.fit(x_train,y_train)

# y_pred = xgb_model.predict(x_test)

# accuracy = accuracy_score(y_test,y_pred)

# print(accuracy)