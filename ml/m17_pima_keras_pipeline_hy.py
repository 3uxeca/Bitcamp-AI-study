from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy
import tensorflow as tf 

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드 
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",") # (.)현재폴더, (..)상위폴더
X = dataset[:, 0:8] # 당뇨병 여부를 알아보기 위한 나이, 가족력 등의 지표 8개
Y = dataset[:,8] # 당뇨병이 있냐 없냐를 0, 1로 분류(마지막 9열)

# scaler 사용
scaler = MinMaxScaler() # 0.7532
# scaler = StandardScaler() # 0.7273
scaler.fit(X)
X_scaled = scaler.transform(X)

from sklearn.model_selection import train_test_split # 사이킷런의 분할기능
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, random_state=66, test_size=0.4 # test를 40%로 train을 60%의 양으로 분할
)

# 모델의 설정
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # sigmoid는 결과값이 무조건 0 or 1

# 모델 컴파일
model.compile(loss='binary_crossentropy', # 이진분류 sigmoid에서 사용하는 loss. softmax에서는 categorical_crossentropy 
              optimizer='adadelta',
              metrics=['accuracy'])

# 모델 실행
model.fit(X_train, Y_train, epochs=200, batch_size=50)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
# Accuracy : 0.7904