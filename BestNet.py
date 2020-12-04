from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.models import load_model

model=load_model('model.h5')
model.compile(loss='mse', optimizer='rmsprop, metrics=['acc'])
print(model.summary())
hist=model.fit(input, target2, epochs=400, batch_size=20,verbose=1)
print(model.metrics_names)