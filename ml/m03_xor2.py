from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,0]

# 2. 모델
# model = LinearSVC() # 선형분류 ㄴㄴ
# model = SVC() # 2차원을 3차원으로 바꿔서 접어가지고 값들을 분류함
model = KNeighborsClassifier(n_neighbors=1) # 가까운이웃 한개!!! 여러 수 넣어서 acc가 잘나오면 좋은 수!

# 3. 실행
model.fit(x_data, y_data)

# 4. 평가 예측

x_test = [[0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)
print("acc = ", accuracy_score([0,1,1,0], y_predict)) # accuracy_score(원래값, 비교값)