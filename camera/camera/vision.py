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
from geometry_msgs.msg import Pose
from math import dist
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Int16
from ament_index_python.packages import get_package_share_path
from tqdm.auto import tqdm
import matplotlib.patches
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

class State(Enum):
    """The current state of the scan."""
    IDLE = auto()
    SCANNING = auto()


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

    def __init__(self, color_image):
        # CLIP related parameters
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.patch_size = 64
        self.window_size = 3
        self.threshold = 0.8
        self.color_img = color_image
    
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
        for _ in range(3):
            scores = np.clip(scores-scores.mean(), 0, np.inf)
        # normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores
    
    def get_box(self, scores):
        detection = scores > self.threshold
        # find box corners
        y_min, y_max = np.nonzero(detection)[:,0].min().item(), np.nonzero(detection)[:,0].max().item()+1
        x_min, x_max = np.nonzero(detection)[:,1].min().item(), np.nonzero(detection)[:,1].max().item()+1
        # convert from patch co-ords to pixel co-ords
        y_min *= self.patch_size
        y_max *= self.patch_size
        x_min *= self.patch_size
        x_max *= self.patch_size
        # calculate box height and width
        height = y_max - y_min
        width = x_max - x_min
        return x_min, y_min, width, height
    
    def detect(self, prompts, stride=1):
        # build image patches for detection
        img_patches = self.get_patches()
        # convert image to format for displaying with matplotlib
        image = np.moveaxis(self.color_img.data.numpy(), 0, -1)
        # initialize plot to display image + bounding boxes
        # fig, ax = plt.subplots(figsize=(Y*0.5, X*0.5))
        # ax.imshow(image)
        # return a list of x, y, width, height, and prompt for each bounding box
        boxes = []
        # process image through object detection steps
        for i, prompt in enumerate(tqdm(prompts)):
            scores = self.get_scores(img_patches, prompt)
            x, y, width, height = self.get_box(scores)
            boxes.append(BoundingBox(x, y, width, height, prompt))
            """
            # create the bounding box
            rect = matplotlib.patches.Rectangle((x, y), width, height, linewidth=3, edgecolor=colors[i], facecolor='none')
            # add label of bounding box to plot
            ax.text(x, y, prompt, fontsize=12, color=colors[i])
            # add the patch to the Axes
            ax.add_patch(rect)
            """
        # plt.show()
        return boxes


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
        # create cv bridge
        self.bridge = CvBridge()
        # initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # set current state
        self.state = State.SCANNING

        # initialize color and depth images
        self.color = None
        self.depth = None

        # initialize intrinsics
        self.intrinsics = None


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
    
    def scan(self):
        # take self.color to do CLIP
        # convert self.color to a tensor of shape [3, height, width]
        self.get_logger().info("I'm here 1")
        color_tensor = torch.tensor(self.color).permute(2, 0, 1).unsqueeze(0).float()
        # get rid of the first dimension of color_tensor
        color_tensor = color_tensor.squeeze(0)
        # log the shape of the color tensor
        # self.get_logger().info(f"Color tensor shape: {color_tensor.shape}")
        # initialize a CLIP model
        clip_model = CLIP(color_tensor)
        # declare prompts
        prompts = ["a laptop", "a computer mouse", "a keyboard", "a balloon"]
        self.get_logger().info("I'm here 2")
        bounding_boxes = clip_model.detect(prompts)
        self.get_logger().info("I'm here 3")
        # log the bounding boxes
        for box in bounding_boxes:
            x, y, width, height, prompt = box.get_attributes()
            self.get_logger().info(f"x: {x}, y: {y}, width: {width}, height: {height}, prompt: {prompt}")
        self.get_logger().info("I'm here 4")

    def timer_callback(self):
        # log state
        self.get_logger().info(f"State: {self.state}")
        if self.state == State.IDLE:
            return
        elif self.state == State.SCANNING:
            if self.color is None or self.depth is None:
                return
            self.scan()
            self.state = State.IDLE

def main(args=None):
    """Start and spin the node."""
    rclpy.init(args=args)
    node = Vision()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()