import keras
from keras.models import *
from keras.layers import *
import numpy as np

x_data = np.arange(1,101)
x_test = np.arange(101,110)
y_test = np.array([107,108,109,110])
###예측에 쓰이는 기간
size = 6
pre_day = 1

##TRAIN 데이터와 LABEL데이터를 나누는 곳
def split(seq, size):
    aaa=[]
    for i in range(len(seq)-size):
        subset = seq[i:(i+size)]
        aaa.append(subset)

    
    return np.array(aaa)
def split_test(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)

    
    return np.array(aaa)
def split_label(seq, size,pre_day):
    lab = []
    lab = np.array(lab)
    if pre_day-1 != 0:
        lab = seq[size:-pre_day+1]
    else:
        lab = seq[size:]

    print(lab.shape)
    
    for i in range(1,pre_day):
        
        # print("%d"%i,seq[size+i:-pre_day+(i+1)].shape)
        if (pre_day-1)!= i:
            lab = np.c_[lab[:], seq[size+i:-pre_day+(i+1)]]
        else:
            lab = np.c_[lab[:], seq[size+i:]]
        print(lab.shape)
    return lab

x_train = split(x_data,size)
y_train = split_label(x_data,size,pre_day)
print(x_train.shape)
print(y_train)
x_test = split_test(x_test,size)
print(x_train)
print(x_test)
x_train = x_train.reshape((-1,6,1))
x_test = x_test.reshape((-1,6,1))
model = Sequential()

model.add(LSTM(36,input_shape=(6,1)))


model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(6))
model.add(Dense(1))


keras.optimizers.Adam(lr=0.5)
rmse= 50
model.compile(loss="mse",optimizer="adam",metrics=["mae"])
while rmse > 0.1:
    


    early_stoping_callback = keras.callbacks.EarlyStopping(monitor="loss",patience=10)

    history = model.fit(x_train, y_train,epochs = 100, batch_size=10, verbose=2,callbacks=[early_stoping_callback])

    print("\n test acc: %.4f"%(model.evaluate(x_train, y_train)[1]))






    y_ = model.predict(x_test)


    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import mean_absolute_error
    def RMSE(y_test, y_):
        return np.sqrt(mean_squared_error(y_test,y_))
    def RMAE(y_test, y_):
        return np.sqrt(mean_absolute_error(y_test,y_))
    rmse = RMSE(y_test,y_)
    print("RMSE:",RMSE(y_test,y_))
    
y_ = model.predict(x_test)
print(y_)