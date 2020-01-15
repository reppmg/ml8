import math

import numpy
import numpy as np
from keras.layers import Dense
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
# train autoregression
model = AR(y[0:len(y)//2])
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=dynamic)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(x_test, test)
pyplot.plot(x_test, predictions, color='red')


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = np.reshape(y, (y.shape[0], 1))
dataset = scaler.fit_transform(dataset)

# reshape into X=t and Y=t+1
look_back = 11
train, test = dataset[0:train_size, :], dataset[train_size - look_back:len(dataset), :]
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(4, input_shape=(1, look_back)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
# lstm_model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# lstm_prediction = lstm_model.predict(testX)
# lstm_prediction = scaler.inverse_transform(lstm_prediction)
# pyplot.plot(x_test[0:len(lstm_prediction)], lstm_prediction, color='green')

pyplot.title("Ongoing " + ("dynamic" if dynamic else "static"))
pyplot.legend(["expected", "AR", "LSTM"])
pyplot.show()
