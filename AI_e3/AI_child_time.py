from __future__ import print_function

# import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping
import pandas as pd

epochs = 10
batch_size = 16

train_df = pd.read_excel('data_sets/ex2_data_set.xlsx')
test_df = pd.read_excel('data_sets/ex2_test_set.xlsx')

# Creates the test arrays
test_X = test_df.drop(columns=['children'])
test_y = test_df[['children']]

#create a dataframe with all training data except the target column
train_X = train_df.drop(columns=['children'])

#create a dataframe with only the target column
train_y = train_df[['children']]

#get number of columns in training data
n_cols = train_X.shape[1]


# BEGIN - create model

# #create model
# model = Sequential()
#
#
# #add model layers
# model.add(Dense(256, activation='relu', input_shape=(n_cols,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='linear'))
#
# #compile model using mse as a measure of model performance
# model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])
#
# #set early stopping monitor so the model stops training when it won't improve anymore
# # early_stopping_monitor = EarlyStopping(monitor='loss', patience=.8)
# #train model
# model.fit(train_X, train_y,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(test_X, test_y),
#           # callbacks=[early_stopping_monitor]
#           )
#
# test_y_predictions = model.predict(test_X)
#
# # for i in range(test_y_predictions)
# #     print("Predicted: ", test_y_predictions[i], " actual: ", test_y[i])
# print("Predicted children: ", test_y_predictions)
# print("Actual children: ", test_y[['children']])

# END - create model

# BEGIN - save json and create model

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
#
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# END - save json and create model


# You can run just the following part as the model was already trained and saved locally
# BEGIN - load json and create model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")



# evaluate loaded model on test data
loaded_model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(test_X, test_y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
print(test_X)
# test_data = {'age': [26], 'married_for': [0], 'income': [9], 'zone': [1]}
test_data = {'age': [28], 'married_for': [0], 'income': [6], 'zone': [2]}
df = pd.DataFrame(test_data)
print(df)
test_y_predictions = loaded_model.predict(df)
print("Predicted children: ", test_y_predictions)
print("Actual children: ", 0)
# print("Actual children: ", test_y[['children']])

# END - load json and create model

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