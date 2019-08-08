import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

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
model = XGBClassifier()

def create_hyperparameters():
    n_estimators=[10, 50, 100]
    learning_rate=[0.1, 0.5, 1.0, 1.5]
    max_depth=[1,2,3]
    random_state=[0,1,2,3]
    eval_metric=['error', 'rmse', 'map']
    base_score=[0.4, 0.6, 0.8]
    early_stopping_rounds=[1, 10, 50]
    n_jobs=[1,2,3,-1]
    return{"n_estimators":n_estimators, "learning_rate":learning_rate, 
            "max_depth":max_depth, "random_state":random_state,
            "eval_metric":eval_metric, "base_score":base_score,
            "early_stopping_rounds":early_stopping_rounds, "n_jobs":n_jobs}

hyperparameters = create_hyperparameters()    

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

seed = 77
n_splits = 3
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
m25_XGB.py
n_jobs = 1
0.9392547217968351
{'random_state': 3, 'n_jobs': -1, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 1.0, 'eval_metric': 'rmse', 'early_stopping_rounds': 10, 'base_score': 0.8}
n_jobs = 2
0.9392547217968351
{'random_state': 1, 'n_jobs': -1, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 1.0, 'eval_metric': 'rmse', 'early_stopping_rounds': 1, 'base_score': 0.8}
n_jobs = 3
0.9392547217968351
{'random_state': 3, 'n_jobs': 2, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 1.0, 'eval_metric': 'rmse', 'early_stopping_rounds': 10, 'base_score': 0.4}
n_jobs = -1
0.9392547217968351
{'random_state': 0, 'n_jobs': 1, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 1.0, 'eval_metric': 'error', 'early_stopping_rounds': 10, 'base_score': 0.6}
'''
