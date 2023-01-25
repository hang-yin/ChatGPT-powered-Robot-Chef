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

class State(Enum):
    """The current state of the scan."""
    IDLE = auto()
    SCANNING = auto()
    DONE = auto()

class Vision(Node):
    """
    TODO: Add class docstring
    """

    def __init__(self):
        super().__init__('vision')
        # set timer frequency
        self.frequency = 60
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
    
    def timer_callback(self):
        pass

def main(args=None):
    """Start and spin the node."""
    rclpy.init(args=args)
    node = Vision()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()