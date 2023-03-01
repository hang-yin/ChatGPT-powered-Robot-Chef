import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pyrealsense2 as rs

model = load_model('hand_activity_model.h5')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

actions = ['grabbing', 'cutting']
colors = [(245,117,16), (117,245,16), (16,117,245)]
threshold = 0.8

def extract_keypoints(self, results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def prob_viz(res, action, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if (prob > 0.5):
            cv2.rectangle(output_frame, (0,60+num*40), (150, 90+num*40), colors[num], -1)
        else:
            cv2.rectangle(output_frame, (0,60+num*40), (0, 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def main():

    pose = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # create realsense pipeline
    pipeline = rs.pipeline()

    # width, height = 640, 480
    width, height = 1280, 720

    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    prev_frame_time = 0

    sequence = []
    sentence = []
    predictions = []
    sequence_length = 45

    try:
        while True: 
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print('No color frame')
                continue
            image = np.asanyarray(color_frame.get_data())
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            hand_result = results.multi_hand_landmarks
            if not hand_result:
                continue

            result_np_array = np.zeros((1, 63))
            result_np_array[0] = np.array([[res.x, res.y, res.z] for res in hand_result[0].landmark]).flatten()

            result_np_array = result_np_array.reshape(63) # 126 np array

            for hand_landmarks in hand_result:
                mp_drawing.draw_landmarks(image, 
                                            hand_landmarks,
                                            mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
            print(result_np_array)
            sequence.append(result_np_array)
            sequence = sequence[-sequence_length:]
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print out probabilities for each action
                # print(res)
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold:     
                        if len(sentence) > 0: 
                                if self.actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(self.actions[np.argmax(res)])
                                else:
                                    sentence.append(self.actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions[np.argmax(res)], actions, image, colors)
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        pose.close()
        pipeline.stop()


if __name__ == "__main__":
    main()