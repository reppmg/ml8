import math
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import numpy as np
from utils import lcm

a = [13, 11, 6]
b = [3, 4, 11]  # b/2pi
c = [14.88, 1.88, 7.22]
T = 2 * math.pi
print(T )


def fun(x):
    result = [a[i] * math.sin(b[i] * x + c[i]) for i in range(len(a))]
    return sum(result)


x = np.arange(0, 2 * T, 0.1)
y = np.array([fun(i) for i in x])
print(x)
print(y)

x_train, x_test = x[1:len(x) // 2], x[len(x) // 2:]
train, test = y[1:len(y) // 2], y[len(y) // 2:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(x_test, test)
pyplot.plot(x_test, predictions, color='red')
pyplot.show()
