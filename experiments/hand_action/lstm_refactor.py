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
import pyrealsense2 as rs

class HandTracking():
    def __init__(self,
                 data_path='data',
                 no_sequence=60,
                 sequence_length=45,
                 start_folder=1,
                 actions=['grabbing', 'cutting', 'else']):
        self.data_path = os.path.join(data_path)
        # Thirty videos worth of data
        self.no_sequence = no_sequence
        # Videos are going to be 30 frames in length
        self.sequence_length = sequence_length
        # Starting from this index
        self.start_folder = start_folder
        # List of actions
        self.actions = actions
        self.label_map = {label:num for num, label in enumerate(actions)}
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

    def realsense_setup(self):
        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # self.config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
    
    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_styled_landmarks(self, image, results):
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image,
                                       results.left_hand_landmarks,
                                       self.mp_holistic.HAND_CONNECTIONS, 
                                       self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                       self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                      ) 
        # Draw right hand connections  
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                       self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                       self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                      )

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])
    
    def collect_data(self, action):
        file_list = os.listdir(os.path.join(self.data_path, action))
        if len(file_list) > 0:
            dirmax = np.max(np.array(file_list).astype(int))
        else:
            dirmax = 0
        for sequence in range(1,self.no_sequence+1):
            try: 
                os.makedirs(os.path.join(self.data_path, action, str(dirmax+sequence)))
            except:
                pass
        
        try:
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                # Loop through sequences aka videos
                for sequence in range(self.start_folder, self.start_folder+self.no_sequence):
                    # Loop through video length aka sequence length
                    frame_num = 0
                    while frame_num < self.sequence_length:
                        # Read feed
                        got_frame = False
                        while not got_frame:
                            frames = self.pipeline.wait_for_frames()
                            depth_frame = frames.get_depth_frame()
                            color_frame = frames.get_color_frame()
                            if depth_frame and color_frame:
                                got_frame = True
                        
                        # show color_frame
                        cv2.imshow('color_frame', np.asanyarray(color_frame.get_data()))
                        
                        # Convert images to numpy arrays
                        depth_image = np.asanyarray(depth_frame.get_data())
                        color_image = np.asanyarray(color_frame.get_data())

                        # Make detections
                        image, results = self.mediapipe_detection(color_image, holistic)
                        # if no hand detected, run this frame again
                        if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
                            print('No hand detected')
                            continue

                        # Draw landmarks
                        self.draw_styled_landmarks(image, results)
                        
                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (100,100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (25,25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (25,25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(self.data_path, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)
                        frame_num += 1

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
        finally:
            # Stop streaming
            self.pipeline.stop()
    
    def data_preprocessing(self):
        sequences, labels = [], []
        for action in self.actions:
            for sequence in np.array(os.listdir(os.path.join(self.data_path, action))).astype(int):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.data_path, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1)
    
    def train_model(self, epochs):
        log_dir = os.path.join("Logs")
        tb_callback = TensorBoard(log_dir=log_dir)

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length,126)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(self.actions), activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.fit(self.X_train, self.y_train, epochs=epochs, callbacks=[tb_callback])
        self.model.summary()
        # save model
        self.model.save('hand_activity_model.h5')
    
    def evaluate_model(self):
        self.model.evaluate(self.X_test, self.y_test)
    
    def load_model(self):
        self.model.load_model('hand_activity_model.h5')

    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame
    
    def predict_in_real_time(self):
        # 1. New detection variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                self.draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-self.sequence_length:]
                
                if len(sequence) == self.sequence_length:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(self.actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
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
                    image = self.prob_viz(res, self.actions, image, self.colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    
    def show_example_video(self, action, sample):
        for i in range(1, self.sequence_length+1):
            res = np.load(os.path.join(self.data_path, action, str(sample), "{}.npy".format(i)))
            print(res.shape)
            # reshape the array to 42x3
            res = res.reshape(42, 3)
            # plot the 21 3D keypoints for both hands in matplotlib
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(42):
                ax.scatter(res[i, 0], res[i, 1], res[i, 2])
            plt.show()
            

if __name__ == "__main__":
    hand_activity = HandTracking()
    hand_activity.realsense_setup()
    hand_activity.data_preprocessing()
    hand_activity.train_model(epochs=10)
    hand_activity.evaluate_model()
    # hand_activity.collect_data('grabbing')
    # hand_activity.show_example_video('grabbing', 10)
    """
    # Load the npy file
    my_array = np.load('data/grabbing/20/0.npy')

    # Print the loaded array
    print(my_array)
    """


                