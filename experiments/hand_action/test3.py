import argparse
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import pyrealsense2 as rs
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

action = 'else'

def extract_keypoints(self, results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

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

    file_list = os.listdir(os.path.join('data', action))
    if len(file_list) > 0:
        dirmax = np.max(np.array(file_list).astype(int))
    else:
        dirmax = 0
    for sequence in range(1,60+1):
        try: 
            os.makedirs(os.path.join('data', action, str(dirmax+sequence)))
        except:
            pass

    try:
        for sequence in range(1, 60+1):
            frame_num = 0
            while frame_num < 45:

                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print('No color frame')
                    continue

                image = np.asanyarray(color_frame.get_data())

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                hand_result = results.multi_hand_landmarks
                
                # check length of hand_result
                if not hand_result:
                    continue
                    
                # check length of hand_result
                # if it's 1, then attach numpy zeros so that it contains 2 elements
                # if it's 2, then do nothing
                result_np_array = np.zeros((2, 63))
                if len(hand_result) == 1:
                    result_np_array[0] = np.array([[res.x, res.y, res.z] for res in hand_result[0].landmark]).flatten()
                elif len(hand_result) == 2:
                    result_np_array[0] = np.array([[res.x, res.y, res.z] for res in hand_result[0].landmark]).flatten()
                    result_np_array[1] = np.array([[res.x, res.y, res.z] for res in hand_result[1].landmark]).flatten()

                # reshape the result_np_array from (2,63) to (126)
                result_np_array = result_np_array.reshape(126)

                for hand_landmarks in hand_result:
                    mp_drawing.draw_landmarks(image, 
                                              hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (100,100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (25,25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (25,25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                npy_path = os.path.join('data', action, str(sequence), str(frame_num))
                np.save(npy_path, result_np_array)

                frame_num += 1

                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        pose.close()
        pipeline.stop()


if __name__ == "__main__":
    main()