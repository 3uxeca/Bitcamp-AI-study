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
model = RandomForestClassifier(n_estimators=300, # 생성할 decision tree의 개수
                               random_state=200,
                               class_weight='balanced_subsample',
                               warm_start=True,
                               oob_score=True,
                               verbose=1,
                               max_features=5, # 최대 선택할 특성의 수
                               n_jobs=-1)    
# model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                 max_depth=2, max_features='auto', max_leaf_nodes=None,
#                 min_impurity_decrease=0.0, min_impurity_split=None,
#                 min_samples_leaf=1, min_samples_split=2,
#                 min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
#                 oob_score=True, random_state=0, verbose=0, warm_start=True)                           
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred)) # 예측 결과 report 출력
print("정답률=", accuracy_score(y_test, y_pred)) 
print(aaa) #(0.9469)