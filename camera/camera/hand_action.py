import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from enum import Enum, auto
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import cv2

class State(Enum):
    """The current state of the scan."""
    IDLE = auto()
    SCANNING = auto()

class HandActionPrediction():
    def __init__(self):
        self.model = load_model('hand_activity_model.h5')
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.actions = ['grabbing', 'cutting']
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = 0.8
    
    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    def prob_viz(self, res, action, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            if (prob > 0.5):
                cv2.rectangle(output_frame, (0,60+num*40), (150, 90+num*40), colors[num], -1)
            else:
                cv2.rectangle(output_frame, (0,60+num*40), (0, 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame
    
    def predict(self, frame):
        with self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                keypoints = self.extract_keypoints(results)
                keypoints = keypoints.reshape(1, 42)
                res = self.model.predict(keypoints)
                if res[0][0] > self.threshold:
                    return 0
                elif res[0][1] > self.threshold:
                    return 1
                else:
                    return -1
            else:
                return -1

class HandActionNode(Node):
    """
    This node will run a state machine
    It will first wait for a message from the motion node
        that tells this node to start looking for hand action
    Once it starts looking for hand action, 
        it will send messages to the motion node
        to tell it what the person is doing
    """
    def __init__(self):
        super().__init__('hand_action_node')
        self.frequency = 10
        self.state = State.IDLE

        # create cv bridge
        self.bridge = CvBridge()

        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # create subscriber for a Bool from /start_action_scan topic
        self.start_action_scan_sub = self.create_subscription(Bool,
                                                              '/start_action_scan',
                                                              self.start_action_scan_callback,
                                                              10)

        # create publisher for a Bool to /hand_action topic
        # 0 indicates grabbing, 1 indicates cutting
        self.hand_action_pub = self.create_publisher(Bool, '/hand_action', 10)

        # create subscriptions to camera color, camera depth, camera aligned color and depth
        self.color_sub = self.create_subscription(Image,
                                                  '/camera/color/image_raw',
                                                  self.color_callback,
                                                  10)
        self.depth_sub = self.create_subscription(Image,
                                                  '/camera/aligned_depth_to_color/image_raw',
                                                  self.depth_callback,
                                                  10)
        self.info_sub = self.create_subscription(CameraInfo,
                                                 '/camera/aligned_depth_to_color/camera_info',
                                                 self.info_callback,
                                                 10)

        # initialize color and depth images
        self.color = None
        self.depth = None

        # initialize intrinsics
        self.intrinsics = None

    def start_action_scan_callback(self, msg):
        """Callback for the start_action_scan subscriber."""
        if msg.data:
            self.get_logger().info('Starting hand action scan')
            self.state = State.SCANNING
        else:
            self.get_logger().info('Stopping hand action scan')
            self.state = State.IDLE
    
    def timer_callback(self):
        """Callback for the timer."""
        if self.state == State.SCANNING:
            self.get_logger().info('Scanning for hand action')
        if self.state == State.IDLE:
            return

def main(args=None):
    """Start and spin the node."""
    rclpy.init(args=args)
    node = HandActionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
