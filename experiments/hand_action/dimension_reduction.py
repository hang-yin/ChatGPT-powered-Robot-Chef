import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X_train = None
X_test = None
y_train = None
y_test = None

# loading data
# actions=['grabbing', 'cutting', 'else']
actions=['grabbing', 'cutting']
data_path='data'
no_sequence=60
sequence_length=45
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(data_path, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
print(X.shape)

"""
# x shape is (180, 45, 126)
# cut half of the data so that x shape is (180, 45, 63)
X = X[:, :, :63]

# define euclidean distance function that takes in 3D points
def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

# the third dimension of X is 63, which is 21 3D points(xyz coordinates)
# we want to calculate the euclidean distance of all 20 points to the first point(wrist)
# so we need to reshape the third dimension to 20 3D points

# reshape X to (180, 45, 21, 3)
X = X.reshape(X.shape[0], X.shape[1], 21, 3)

Xnew = np.zeros((X.shape[0], X.shape[1], 21))

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(21):
            if (k < 20):
                Xnew[i, j, k] = euclidean_distance(X[i, j, 0, 0], X[i, j, 0, 1], X[i, j, 0, 2], X[i, j, k+1, 0], X[i, j, k+1, 1], X[i, j, k+1, 2])
            else:
                # the last data point is the distance from the current wrist point to the previous wrist point
                # if this is the first wrist point, then the distance is 0
                if (j == 0):
                    Xnew[i, j, k] = 0
                else:
                    Xnew[i, j, k] = euclidean_distance(X[i, j, 0, 0], X[i, j, 0, 1], X[i, j, 0, 2], X[i, j-1, 0, 0], X[i, j-1, 0, 1], X[i, j-1, 0, 2])

X = Xnew
"""
print(X.shape)

y = to_categorical(labels).astype(int)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)





log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=20, callbacks=[tb_callback])
model.summary()
# save model
model.save('hand_activity_model.h5')

model.evaluate(X_test, y_test)