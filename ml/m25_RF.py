import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1) # "quality"라는 column만 drop하고 나머지는 x값이 된다.

# y 레이블 변경하기 --- (*2)
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

# 학습 데이터와 평가 데이터로 분리하기 
x_train, x_test, y_train, y_test = train_test_split(
                                    x, y, test_size=0.2)

# 학습하기 
model = RandomForestClassifier()

def create_hyperparameters():
    max_depth=[1, 10, 50]
    n_estimators=[10,50,100]
    criterion=['gini', 'entropy']
    class_weight=[None, 'balanced', 'balanced_subsample']
    n_jobs=[1,2,3,-1]
    return{"max_depth":max_depth, "n_estimators":n_estimators, "criterion":criterion, 
            "class_weight":class_weight, "n_jobs":n_jobs}

hyperparameters = create_hyperparameters()    

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

seed = 77
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=77)

for num_jobs in range(-1, 4):
    if num_jobs == 0:
         continue
    search = RandomizedSearchCV(estimator=model,
                                 param_distributions=hyperparameters,
                                 n_iter=10, n_jobs=num_jobs, cv=kfold, verbose=1)
    search.fit(x_train, y_train) # 데이터 집어넣기!

    print(search.best_score_)
    print(search.best_params_)
    print("--"*15)
'''
m25_RF.py
n_jobs = 1
0.9407861153649821
{'n_jobs': -1, 'n_estimators': 100, 'max_depth': 50, 'criterion': 'gini', 'class_weight': 'balanced_subsample'}
n_jobs = 2
0.9415518121490556
{'n_jobs': 1, 'n_estimators': 50, 'max_depth': 50, 'criterion': 'gini', 'class_weight': None}
n_jobs = 3
0.9418070444104135
{'n_jobs': -1, 'n_estimators': 50, 'max_depth': 50, 'criterion': 'gini', 'class_weight': None}
n_jobs = -1
0.9420622766717713
{'n_jobs': 2, 'n_estimators': 100, 'max_depth': 50, 'criterion': 'entropy', 'class_weight': None}
'''