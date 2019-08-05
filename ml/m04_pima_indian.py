from keras.models import Sequential
from keras.layers import Dense
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

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # sigmoid는 결과값이 무조건 0 or 1

# 모델 컴파일
model.compile(loss='binary_crossentropy', # 이진분류 sigmoid에서 사용하는 loss. softmax에서는 categorical_crossentropy 
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

# Accuracy : 0.7904