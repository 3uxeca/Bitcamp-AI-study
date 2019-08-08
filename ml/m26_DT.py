from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# 기온 데이터 읽어 들이기
df = pd.read_csv('./data/tem10y.csv', encoding="utf-8")

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

# 6개씩 잘라서 사용한 시계열 => LSTM으로 바꿀 수 있다.

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = [] # 학습 데이터
    y = [] # 결과
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

# 학습
model = DecisionTreeRegressor()

def create_hyperparameters():
    splitter = ["best", "random"]
    min_samples_split = [0.1, 0.2, 0.5, 1.0]
    max_features = [None, 30, "auto", "sqrt", "log2"]
    max_depth = [None, 1, 2, 3]
    criterion = ['mse', 'friedman_mse', 'mae']
    random_state = [None, 0, 1, 2]
    return{'splitter':splitter, 'random_state':random_state,
            'min_samples_split':min_samples_split, 'max_depth':max_depth, 'criterion':criterion}
      
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

seed = 77
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=77)

for num in range(-1, 4):
    if num == 0:
        continue
    search = RandomizedSearchCV(estimator=model,
                                 param_distributions=hyperparameters,
                                 n_iter=10, n_jobs=num, cv=kfold, verbose=1)

#                              # 작업이 10회 수행, 3겹 교차검증 사용(3조각을 나눠서 검증). n_jobs는 알아서 찾아볼 것.
# KFold가 5번 돌았다면 얘는 랜덤하게 돈다. 이 작업을 하는 것은 위의 하이퍼파라미터 중 최적의 결과를 내는 파라미터들을 찾기 위함.
# search.fit(data["x_train"], data["y_train"])

    search.fit(train_x, train_y) # 데이터 집어넣기!

# R2(결정계수) 구하기
    from sklearn.metrics import r2_score
    r2 = r2_score(search.predict(test_x), test_y)

    print("R2 : ", r2)
    print(search.best_params_)
    print("-"*40)

'''
1. m26_DT.py
n_jobs = 1
R2 :  0.8829783194613108
{'splitter': 'random', 'random_state': 2, 'min_samples_split': 0.2, 'max_depth': None, 'criterion': 'friedman_mse'}
n_jobs = 2
R2 :  0.8835558607381748
{'splitter': 'best', 'random_state': 2, 'min_samples_split': 0.1, 'max_depth': 3, 'criterion': 'friedman_mse'}
n_jobs = 3
R2 :  0.8928663299930102
{'splitter': 'random', 'random_state': 1, 'min_samples_split': 0.1, 'max_depth': None, 'criterion': 'friedman_mse'}
n_jobs = -1
R2 :  0.8953754265421827
{'splitter': 'best', 'random_state': 2, 'min_samples_split': 0.1, 'max_depth': None, 'criterion': 'mse'}
'''