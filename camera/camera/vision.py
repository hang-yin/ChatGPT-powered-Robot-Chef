import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pyrealsense2 as rs2
from enum import Enum, auto
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Point
from math import dist
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Int16, Int64
from ament_index_python.packages import get_package_share_path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel
from plan_execute_interface.msg import DetectedObject
import mediapipe as mp
from tensorflow.keras.models import load_model

class State(Enum):
    """The current state of the scan."""
    IDLE = auto()
    SCANNING = auto()
    PUBLISH_OBJECTS = auto()
    ACTION_SCAN = auto()
    FIND_TABLE = auto()


class BoundingBox():
    """A bounding box for an object."""
    def __init__(self, x, y, w, h, prompt):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.prompt = prompt
    
    def get_attributes(self):
        return self.x, self.y, self.w, self.h, self.prompt


class CLIP():
    """
    TODO: Add class docstring
    """

    def __init__(self, color_image, node):
        # CLIP related parameters
        model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.patch_size = 32
        self.window_size = 3
        self.threshold = 0.8
        self.color_img = color_image
        self.node = node
    
    def get_patches(self):
        # add extra dimension for later calculations
        img_patches = self.color_img.data.unfold(0,3,3)
        # break the image into patches (in height dimension)
        img_patches = img_patches.unfold(1, self.patch_size, self.patch_size)
        # break the image into patches (in width dimension)
        img_patches = img_patches.unfold(2, self.patch_size, self.patch_size)
        return img_patches
    
    def get_scores(self, img_patches, prompt, stride=1):
        # initialize scores and runs arrays
        scores = torch.zeros(img_patches.shape[1], img_patches.shape[2])
        runs = torch.ones(img_patches.shape[1], img_patches.shape[2])
        # iterate through patches
        for Y in range(0, img_patches.shape[1]-self.window_size+1, stride):
            for X in range(0, img_patches.shape[2]-self.window_size+1, stride):
                # initialize array to store big patches
                big_patch = torch.zeros(self.patch_size*self.window_size, self.patch_size*self.window_size, 3)
                # get a single big patch
                patch_batch = img_patches[0, Y:Y+self.window_size, X:X+self.window_size]
                # iteratively build all big patches
                for y in range(self.window_size):
                    for x in range(self.window_size):
                        big_patch[y*self.patch_size:(y+1)*self.patch_size, x*self.patch_size:(x+1)*self.patch_size, :] = patch_batch[y, x].permute(1, 2, 0)
                
                # if the mean of corresponding depth patch is greater than the table height, skip the patch
                # if self.depth_img.data[Y:Y+self.window_size, X:X+self.window_size].mean() > self.table_height:
                    # continue

                inputs = self.processor(
                    images=big_patch, # image trasmitted to the model
                    return_tensors="pt", # return pytorch tensor
                    text=prompt, # text trasmitted to the model
                    padding=True
                ).to(self.device) # move to device if possible

                score = self.model(**inputs).logits_per_image.item()
                # sum up similarity scores
                scores[Y:Y+self.window_size, X:X+self.window_size] += score
                # calculate the number of runs 
                runs[Y:Y+self.window_size, X:X+self.window_size] += 1
        # calculate average scores
        scores /= runs
        # clip scores
        scores = np.clip(scores-scores.mean(), 0, np.inf)
        #for _ in range(3):
        #    scores = np.clip(scores-scores.mean(), 0, np.inf)
        # normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        # create a tensor of zeros with the same shape as scores
        adj_scores = torch.zeros(scores.shape[0], scores.shape[1])
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if scores[i, j] > self.threshold:
                    adj_scores[i, j] = scores[i, j]
        return adj_scores
    
    def get_box(self, scores):
        # detection = scores > self.threshold
        # find box corners
        y_min, y_max = np.nonzero(scores)[:,0].min().item(), np.nonzero(scores)[:,0].max().item()+1
        x_min, x_max = np.nonzero(scores)[:,1].min().item(), np.nonzero(scores)[:,1].max().item()+1
        # convert from patch co-ords to pixel co-ords
        y_min *= self.patch_size
        y_max *= self.patch_size
        x_min *= self.patch_size
        x_max *= self.patch_size
        # calculate box height and width
        height = y_max - y_min
        width = x_max - x_min
        return x_min, y_min, width, height

    def detect(self, prompts):
        # build image patches for detection
        img_patches = self.get_patches()
        # only consider the bottom half of the image
        # img_patches = img_patches[:, img_patches.shape[1]//2:, :, :, :]
        # convert image to format for displaying with matplotlib
        # image = np.moveaxis(self.color_img.data.numpy(), 0, -1)
        # initialize plot to display image + bounding boxes
        # fig, ax = plt.subplots(figsize=(Y*0.5, X*0.5))
        # ax.imshow(image)
        # return a list of x, y, width, height, and prompt for each bounding box
        boxes = []
        # process image through object detection steps
        for prompt in tqdm(prompts):
            # log prompt
            self.node.get_logger().info(f'Processing prompt: {prompt}')
            scores = self.get_scores(img_patches, prompt)
            # log scores
            self.node.get_logger().info(f'Scores: {scores}')
            # log score shape
            self.node.get_logger().info(f'Score shape: {scores.shape}')
            x, y, width, height = self.get_box(scores)
            boxes.append(BoundingBox(x, y, width, height, prompt))
        return boxes

class HandActionPrediction():
    def __init__(self, node):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.actions = ['grabbing', 'cutting']
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = 0.8
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence_length = 45
        self.sequence = []
        self.predictions = []
        self.actions = ['grabbing', 'cutting']
        self.cutting_threshold = 0.47
        self.node = node
    
    def load_model(self, model_path):
        self.model = load_model(model_path)

    def prob_viz(self, prediction, actions, input_frame, colors):
        output_frame = input_frame.copy()
        num = 0
        if (prediction == 0):
            cv2.rectangle(output_frame, (0,60+num*40), (150, 90+num*40), colors[num], -1)
        else:
            cv2.rectangle(output_frame, (0,60+num*40), (0, 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        num = 1
        if (prediction == 1):
            cv2.rectangle(output_frame, (0,60+num*40), (150, 90+num*40), colors[num], -1)
        else:
            cv2.rectangle(output_frame, (0,60+num*40), (0, 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame
    
    def predict(self, frame):
        frame = np.asanyarray(frame)
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = frame
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        """
        hand_result = results.multi_hand_landmarks
        if hand_result:
            result_np_array = np.zeros((1,63))
            result_np_array[0] = np.array([[res.x, res.y, res.z] for res in hand_result[0].landmark]).flatten()
            result_np_array = result_np_array.reshape(63)
            for hand_landmarks in hand_result:
                self.mp_drawing.draw_landmarks(image, 
                                               hand_landmarks,
                                               self.mp_hands.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                               self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
            # print result_np_array
            self.node.get_logger().info(f'Landmarks: {result_np_array}')
            self.sequence.append(result_np_array)
            self.sequence = self.sequence[-self.sequence_length:]
            if len(self.sequence) == self.sequence_length:
                res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                self.node.get_logger().info(f'Predictions: {res}')
                if res[1] > self.cutting_threshold:
                    self.predictions.append(1)
                else:
                    self.predictions.append(0)
                # self.predictions.append(np.argmax(res))
                image = self.prob_viz(self.predictions[-1], self.actions, image, self.colors)
            
            # Show to screen
            cv2.namedWindow('Hand Action', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Hand Action', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                return
        return self.predictions[-1] if len(self.predictions) > 0 else None

class Vision(Node):
    """
    TODO: Add class docstring
    """

    def __init__(self):
        super().__init__('vision')
        # set timer frequency
        self.frequency = 10
        self.timer = self.create_timer(1 / self.frequency, self.timer_callback)
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
        # create publisher for bounding boxes - DetectedObject
        self.bounding_boxes_pub = self.create_publisher(DetectedObject, 'detected_object', 10)

        # create cv bridge
        self.bridge = CvBridge()
        # initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # set current state
        self.state = State.ACTION_SCAN

        # initialize color and depth images
        self.color = None
        self.depth = None
        self.hand_action_color = None

        # initialize intrinsics
        self.intrinsics = None

        self.window = "Bounding boxes on color image"

        # self.prompts = ["eggplant", "carrot", "apple", "yellow pepper"]
        # self.prompts = ["carrot", "green beans", "yellow pepper"]
        # self.prompts = ["banana", "eggplant", "strawberry", "green beans"]
        # self.prompts = ["banana", "strawberry"]
        # self.prompts = ["carrot"]
        # self.prompts = ["chopping board", "yellow pepper", "orange", "kiwi", "green beans", "strawberry", "eggplant", "banana"]
        self.prompts = ["chopping board", "orange", "kiwi", "green beans", "strawberry", "eggplant", "banana"]

        self.object_frame = TransformStamped()
        self.object_frame.header.frame_id = 'camera_link'

        self.detected_objects = []

        # create subscriber for a Bool from /start_action_scan topic
        self.start_action_scan_sub = self.create_subscription(Bool,
                                                              '/start_action_scan',
                                                              self.start_action_scan_callback,
                                                              10)

        # create publisher for a Bool to /hand_action topic
        # 0 indicates grabbing, 1 indicates cutting
        self.hand_action_pub = self.create_publisher(Int64, '/hand_action', 10)

        self.hand_action_classifier = HandActionPrediction(self)

        model_path = get_package_share_path('camera') / 'hand_activity_model.h5'
        self.hand_action_classifier.load_model(model_path)

        self.table_height = 510.0

        # self.height_start_idx = 20
        # self.height_end_idx = 600
        # self.table_area_threshold = 10000
    
    def start_action_scan_callback(self, msg):
        if msg.data:
            self.state = State.ACTION_SCAN
        else:
            self.state = State.IDLE

    def info_callback(self, cameraInfo):
        """Store the intrinsics of the camera."""
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.d]
        except CvBridgeError:
            self.get_logger().info("Getting intrinsics failed?")
            return
    
    def color_callback(self, color):
        """Store the color image."""
        try:
            self.color = self.bridge.imgmsg_to_cv2(color, "bgr8")
            # flip image
            self.color = cv2.flip(self.color, 0)
            self.hand_action_color = self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8')
        except CvBridgeError:
            self.get_logger().info("Getting color image failed?")
            return
    
    def depth_callback(self, depth):
        """Store the depth image."""
        try:
            self.depth = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        except CvBridgeError:
            self.get_logger().info("Getting depth image failed?")
            return

    def draw_bounding_boxes(self, boxes):
        """Draw bounding boxes on the color image using cv2"""
        for box in boxes:
            x, y, width, height, prompt = box.get_attributes()
            cv2.rectangle(self.color, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(self.color, prompt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    def scan(self):
        # take self.color to do CLIP
        # convert self.color to a tensor of shape [3, height, width]
        pil_image = PILImage.fromarray(self.color)
        color_tensor = transforms.ToTensor()(pil_image)
        # switch the first and third channels
        color_tensor = color_tensor[[2, 1, 0], :, :]
        # log the shape of the color tensor
        self.get_logger().info(f"Color tensor shape: {color_tensor.shape}")
        # log the entire color tensor
        self.get_logger().info(f"Color tensor: {color_tensor}")
        """
        # log shape of self.color
        self.get_logger().info(f"Color image shape: {self.color.shape}")
        color_tensor = torch.tensor(self.color).permute(2, 0, 1).float()
        # convert values to be between 0 and 1
        color_tensor = color_tensor / 255
        # log the shape of the color tensor
        self.get_logger().info(f"Color tensor shape: {color_tensor.shape}")
        # log the entire color tensor
        self.get_logger().info(f"Color tensor: {color_tensor}")
        """
        # take self.depth to do CLIP
        # convert self.depth to a tensor with same shape as color_tensor
        # depth_tensor = torch.tensor(self.depth).unsqueeze(0).float()
        # initialize a CLIP model
        # clip_model = CLIP(color_tensor, depth_tensor, self)
        clip_model = CLIP(color_tensor, self)
        # declare prompts
        # prompts = ["a fry pan", "a carrot", "an eggplant"]# , "a computer mouse"] , "a keyboard", "a balloon"]
        # prompts = ["green beans", "a carrot", "an eggplant", "a banana", "an apple", "corn", "a yellow pepper"]
        
        bounding_boxes = clip_model.detect(self.prompts)
        # log the bounding boxes
        for box in bounding_boxes:
            x, y, width, height, prompt = box.get_attributes()
            self.get_logger().info(f"x: {x}, y: {y}, width: {width}, height: {height}, prompt: {prompt}")
            # publish this box
            x_coord = x + int(width / 2)
            y_coord = y + int(height / 2)
            z_coord = self.depth[y_coord, x_coord]
            # log the x, y, z coordinates
            self.get_logger().info(f"Before deprojection x: {x_coord}, y: {y_coord}, z: {z_coord}")
            object_deprojected = rs2.rs2_deproject_pixel_to_point(self.intrinsics,
                                                                  [float(x_coord),
                                                                   float(y_coord)],
                                                                  float(z_coord))
            # log the deprojected point
            self.get_logger().info(f"Deprojected point: {object_deprojected}")
            obj_point = Point(x=object_deprojected[1]/1000.0,
                              y=object_deprojected[0]/1000.0,
                              z=object_deprojected[2]/1000.0)
            # log x, y, z
            self.get_logger().info(f"Sending x: {obj_point.x}, y: {obj_point.y}, z: {obj_point.z} for {prompt}")
            detected_obj = DetectedObject(object_name=prompt,
                                          position=obj_point)
            self.bounding_boxes_pub.publish(detected_obj)
            self.detected_objects.append(detected_obj)
        # show the color image with bounding boxes
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        self.draw_bounding_boxes(bounding_boxes)
        # flip self.color
        cv2.imshow(self.window, self.color)
        cv2.waitKey(0)
    
    """
    def find_largest_contour(self, height_idx):
        depth_cpy = np.array(self.depth)
        # Only keep stuff that's within the appropriate depth band.
        depth_mask = cv2.inRange(np.array(depth_cpy),height_idx,height_idx + 20)
        # This operation helps to remove "dots" on the depth image.
        # Kernel higher dimensional = smoother. It's also less important if camera is farther away.
        # depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, self.kernel)
        # All 0s, useful for following bitwise operations.
        # bounding_mask = np.zeros((self.intrinsics.height, self.intrinsics.width), np.int8)
        # Creating a square over the area defined in self.rect
        # square = cv2.fillPoly(bounding_mask, [self.rect], 255)
        # Blacking out everything that is not within square
        # square = cv2.inRange(square, 1, 255)
        # Cropping the depth_mask so that only what is within the square remains.
        # depth_mask = cv2.bitwise_and(depth_mask, depth_mask) # , mask=square)
        # Find the contours of this cropped mask to help locate tower.
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroids, areas, large_contours = [], [], []
        for c in contours:
            M = cv2.moments(c)
            area = cv2.contourArea(c)
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centroid = (cx, cy)
                if area > 100:
                    centroids.append(centroid)
                    areas.append(area)
                    large_contours.append(c)
            except ZeroDivisionError:
                pass
        largest_area, centroid_pose, = None, None
        max_centroid, box, box_area = None, None, None
        if len(areas) != 0:
            # There is something large in the image.
            largest_index = np.argmax(areas)
            largest_area = areas[largest_index]
            # self.get_logger().info(f"LARGEST AREA: {largest_area}")
            max_centroid = centroids[largest_index]
            max_contour = large_contours[largest_index]
            centroid_depth = depth_cpy[max_centroid[1]][max_centroid[0]]
            centroid_deprojected = rs2.rs2_deproject_pixel_to_point(self.intrinsics,
                                                                    [max_centroid[0],
                                                                     max_centroid[1]],
                                                                    centroid_depth)
            centroid_pose = Pose()
            centroid_pose.position.x = centroid_deprojected[0]/1000.
            centroid_pose.position.y = centroid_deprojected[1]/1000.
            centroid_pose.position.z = centroid_deprojected[2]/1000.

            min_rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(min_rect)
            box = np.intp(box)
            # Save original box area to test if the contour is a good fit
            box_area = dist(box[0], box[1])*dist(box[1], box[2])

        drawn_contours = cv2.drawContours(self.color, large_contours, -1, (0, 255, 0), 3)
        if max_centroid is not None:
            drawn_contours = cv2.circle(drawn_contours, max_centroid, 5, (0, 0, 255), 5)
            drawn_contours = cv2.drawContours(drawn_contours, [box], 0, (255, 0, 0), 3)

        cv2.imshow(self.window, drawn_contours)
        cv2.waitKey(1)
        return largest_area
    """
    def timer_callback(self):
        # log state
        # self.get_logger().info(f"State: {self.state}")
        if self.state == State.IDLE:
            """
            for obj in self.detected_objects:
                self.bounding_boxes_pub.publish(obj)
            # continue to show the color image
            cv2.imshow(self.window, self.color)
            cv2.waitKey(1)
            """
            return
        elif self.state == State.SCANNING:
            if self.color is None or self.depth is None or self.intrinsics is None:
                return
            # cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
            # cv2.imshow(self.window, self.color)
            # cv2.waitKey(0)
            self.scan()
            self.state = State.IDLE
        elif self.state == State.ACTION_SCAN:
            if self.color is None or self.depth is None or self.intrinsics is None:
                return
            prediction = self.hand_action_classifier.predict(self.hand_action_color)
            self.get_logger().info(f"Prediction: {prediction}")
            """
            if prediction is not None:
                # self.get_logger().info(f"Prediction: {prediction}")
                # initialize a Int64 message
                msg = Int64()
                # set the data
                msg.data = int(prediction)
                self.hand_action_pub.publish(msg)
                if prediction == 0:
                    self.get_logger().info("Hand action: grabbing")
                else:
                    self.get_logger().info("Hand action: cutting")
            """
        """
        elif self.state == State.FIND_TABLE:
            if self.color is None or self.depth is None or self.intrinsics is None:
                return
            # find the table
            for i in range(self.height_start_idx, self.height_end_idx):
                self.get_logger().info(f"Scanning at height {i}")
                largest_area = self.find_largest_contour(i)
                self.get_logger().info(f"Area: {largest_area}")
                if largest_area is None:
                    continue
                if largest_area > self.table_area_threshold:
                    self.get_logger().info(f"Found table at height {i}")
                    self.table_height = i
                    self.state = State.SCANNING
                    break
            self.get_logger().info("Failed to find table")
            self.state = State.IDLE   
        """     

def main(args=None):
    """Start and spin the node."""
    rclpy.init(args=args)
    node = Vision()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()