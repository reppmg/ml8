import math

import numpy
import numpy as np
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AR

dynamic = True
a = [13, 11, 6]
b = [3, 4, 11]  # b/2pi
c = [14.88, 1.88, 7.22]
T = 2 * math.pi
print(T)


def fun(x):
    result = [a[i] * math.sin(b[i] * x + c[i]) for i in range(len(a))]
    return sum(result)


test_to_train = 4
x = np.arange(0, test_to_train * T, 0.1)
y = np.array([fun(i) for i in x])
print(x)
print(y)

train_size = len(x) // test_to_train
x_train, x_test = x[1:train_size], x[train_size:]
train, test = y[1:train_size], y[train_size:]

x_train = np.reshape(x_train, (x_train.shape[0], 1))
x_test = np.reshape(x_test, (x_test.shape[0], 1))

model = Sequential()
model.add(Embedding(10000000000, 11))
model.add(LSTM(4, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, train, epochs=100, batch_size=32, verbose=2)
lstm_prediction = model.predict(x_test)
pyplot.plot(x_test, lstm_prediction, color='green')

pyplot.title("Ongoing " + ("dynamic" if dynamic else "static"))
pyplot.legend(["expected", "AR", "LSTM"])
pyplot.show()
