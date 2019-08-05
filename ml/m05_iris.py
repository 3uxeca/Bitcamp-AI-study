import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC   # 분류모델
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8',
                        names=['a', 'b', 'c', 'd', 'y'])
print(iris_data)
# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"] # .loc 레이블로 나누기(레이블은 실질적인 데이터가 아님)
x = iris_data.loc[:,["a", "b", "c", "d"]]

# y2 = iris_data.iloc[:, 4] # .iloc 는 column 기준으로 나누기
# x2 = iris_data.iloc[:, 0:4]

# print("==============================")
# print(x.shape) # (150, 4)
# print(y.shape) # (150,)
# print(x2.shape)
# print(y2.shape)

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8, shuffle=True)

print(x_train.shape) # (120, 4)
print(x_test.shape) # (30,4)
print(y_test) # str 형식. 분류 모델에서 결과값이 str 형식으로 나올 수 있다! 딥러닝에선 one-hot-encodint(categorical_crossentropy)

# 학습하기
# clf = SVC() # 정답률 0.9
# clf = LinearSVC() # 정답률 0.9666
clf = KNeighborsClassifier(n_neighbors=1) # 정답률 0.9666
clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_pred)) 