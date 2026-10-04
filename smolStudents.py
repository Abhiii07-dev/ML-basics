import pandas as pd

data ={ 
    "hours" :[1,2,3,4,5,6],
    "scores" : [52,57,65,70,75,80] }


pd.DataFrame(data).to_csv("smolstudents.csv",index = False)