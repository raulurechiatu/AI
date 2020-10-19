from __future__ import print_function

# import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping
import pandas as pd

epochs = 50
batch_size = 4

train_df = pd.read_excel('employees_data_set 2.xlsx')
test_df = pd.read_excel('employees_test_set 2.xlsx')

# Creates the test arrays
test_X = test_df.drop(columns=['wage_per_hour'])
test_y = test_df[['wage_per_hour']]

#create a dataframe with all training data except the target column
train_X = train_df.drop(columns=['wage_per_hour'])

#create a dataframe with only the target column
train_y = train_df[['wage_per_hour']]

#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#compile model using mse as a measure of model performance
model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore
# early_stopping_monitor = EarlyStopping(monitor='loss', patience=.8)
#train model
model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_X, test_y),
          # callbacks=[early_stopping_monitor]
          )

test_y_predictions = model.predict(test_X)

# for i in range(test_y_predictions)
#     print("Predicted: ", test_y_predictions[i], " actual: ", test_y[i])
print("Predicted wage: ", test_y_predictions)
print("Actual wage: ", test_y[['wage_per_hour']])
# print()

# batch_size = 128
# num_classes = 10
# epochs = 1
#
#
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])