from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

actions = np.array(['hello', 'thanks', 'iloveyou'])

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# create a random X_train and y_train dataset
# X_train should have size (n, 30, 63)
# y_train should have size (n, 1)
# n is the number of samples

X_train = np.random.rand(80, 30, 63)
y_train = np.random.rand(80, 3)

model.fit(X_train, y_train, epochs=10, callbacks=[tb_callback])

model.summary()
