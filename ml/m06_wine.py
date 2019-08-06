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

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y, test_size=0.2)

# 학습하기 
model = RandomForestClassifier(n_estimators=400, # 생성할 decision tree의 개수
                               random_state=44,
                               class_weight='balanced_subsample',
                               warm_start=True,
                               oob_score=True,
                               max_features='auto', # 최대 선택할 특성의 수
                               n_jobs=-1)                               
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred)) # 예측 결과 report 출력
print("정답률=", accuracy_score(y_test, y_pred)) 
print(aaa) #(0.7102)