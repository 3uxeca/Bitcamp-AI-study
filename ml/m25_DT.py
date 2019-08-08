import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

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
model = DecisionTreeClassifier()

def create_hyperparameters():
    class_weight = [None,'balanced']
    max_features = [1, 5, 10]
    max_depth = [1, 2, 3]
    criterion = ['gini', 'entropy']
    random_state = [None, 0, 1, 2]
    return{'class_weight':class_weight, 'random_state':random_state,
            'max_features':max_features, 'max_depth':max_depth, 'criterion':criterion}
      
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
m25_DT.py
n_jobs = 1
0.9280245022970903
{'random_state': 2, 'max_features': 5, 'max_depth': 1, 'criterion': 'gini', 'class_weight': None}
n_jobs = 2
0.9280245022970903
{'random_state': 2, 'max_features': 5, 'max_depth': 1, 'criterion': 'entropy', 'class_weight': None}
n_jobs = 3
0.9280245022970903
{'random_state': 2, 'max_features': 5, 'max_depth': 1, 'criterion': 'gini', 'class_weight': None}
n_jobs = -1
0.9280245022970903
{'random_state': 2, 'max_features': 1, 'max_depth': 1, 'criterion': 'gini', 'class_weight': None}
'''