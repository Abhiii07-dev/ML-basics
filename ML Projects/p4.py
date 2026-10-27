import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\ap910\Downloads\insurance.csv")

#EDA

# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.shape)

# print(df.describe())

# print(df.isnull().sum())



# VISUALSAION

# print(df.columns)


numeric_columns = ["age","bmi","children","charges"]

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde=True,bins = 20)
    # plt.show()





sns.countplot(x = df["children"])
# plt.show()


sns.countplot(x = df["sex"])
# plt.show()


sns.countplot(x = df["smoker"])

# plt.show()



for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x = df[col])
    # plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True) , annot=True)
# plt.show()




# DATA CLEANING AND PREPROCESSING

cleaned_df = df.copy()
# print(cleaned_df.shape)

#drop null values (already done)

cleaned_df.drop_duplicates(inplace = True)
# print(cleaned_df.shape)

#fill the null values

# print(cleaned_df.dtypes)

# print(cleaned_df["sex"].value_counts())



# PREPROCESSING

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cleaned_df["sex"] = le.fit_transform(cleaned_df["sex"])
cleaned_df["smoker"] = le.fit_transform(cleaned_df["smoker"])

cleaned_df.rename(columns={
    "sex" : "is_male",
    "smoker" : "is_smoker"
}, inplace=True)

# print(cleaned_df)


cleaned_df = pd.get_dummies(cleaned_df,columns=["region"],drop_first=True,dtype = int)
# print(cleaned_df)





# FEATURE ENGINEERING

sns.histplot(df["bmi"])
# plt.show()


cleaned_df["bmi_categories"] = pd.cut(
    cleaned_df["bmi"],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels = ["Underweight","normal","overweight","obese"]
)

# print(cleaned_df)

cleaned_df = pd.get_dummies(cleaned_df,columns=["bmi_categories"],drop_first=True,dtype=int)
# print(cleaned_df)

# scaling

from sklearn.preprocessing import StandardScaler



# print(cleaned_df.columns)

col = ["age","bmi","children"]

ss = StandardScaler()

cleaned_df[col] = ss.fit_transform(cleaned_df[col])
# print(cleaned_df)



"FEATURE EXTRACTION"

# PEARSON CORRELATION 
from scipy.stats import pearsonr

selected_features = [
'age', 'is_male', 'bmi', 'children', 'is_smoker', 'charges',
'region_northwest', 'region_southeast', 'region_southwest',
'bmi_categories_normal', 'bmi_categories_overweight',
'bmi_categories_obese'
]

correlations = {
    feature : pearsonr(cleaned_df[feature],cleaned_df['charges'])[0]
    for feature in selected_features

}

correlation_dff = pd.DataFrame(list(correlations.items()),columns = ["feature","Pearson Correlation"])
correlation_dff.sort_values(by = "Pearson Correlation" , ascending=False, inplace=True)

# more the value , more the correlation

# print(correlation_dff)  

# if the value is above 0.5 OR below -0.5 ---> have good correlation ( have to include)
# 0.2 - 0.5 = can include
# 0.1 - 0.2 = negotiable




cat_features = [
    'is_male','is_smoker','region_northwest','region_southeast','region_southwest','bmi_categories_normal','bmi_categories_overweight','bmi_categories_obese'
]

# chi2 test


from scipy.stats import chi2_contingency


alpha = 0.05

cleaned_df["charges_bin"] = pd.qcut(
    cleaned_df["charges"] , q = 4 , labels=False

)
chi2_results  = {}

for col in cat_features:
    contingency = pd.crosstab(cleaned_df[col] , cleaned_df["charges_bin"])
    chi2_stat , p_val , _,_ = chi2_contingency(contingency)
    decision = 'Reject Null (keep feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col]={
        'chi2_statistic' : chi2_stat,
        'p_value' : p_val,
        'Decision' : decision

    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by = 'p_value')
# print(chi2_df)




final_df = cleaned_df[["age","is_male","is_smoker","region_southeast","children","charges"]]


# print("FINAL DATAFRAME:")
# print(final_df)



"SUPERVISED LEARNING"

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = final_df.drop("charges",axis = 1)
y = final_df["charges"]

x_train , x_test , y_train ,y_test = train_test_split(x ,y ,test_size = 0.20 , random_state = 42)


model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# Performance Analysis

from sklearn.metrics import r2_score

r2 = r2_score(y_test , y_pred)
# print(r2)

n = x_test.shape[0]
p = x_test.shape[1]
adjusted_r2  = 1 - ((1-r2)*(n-1)/(n-p-1))
# print(adjusted_r2)

# Ridge and lasso regression if overfitted ot underfitted

