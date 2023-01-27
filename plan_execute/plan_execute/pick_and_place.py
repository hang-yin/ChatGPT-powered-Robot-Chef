import rclpy
from rclpy.node import Node
from enum import Enum, auto
from plan_execute_interface.srv import GoHere, Place
from plan_execute_interface.msg import DetectedObject
from plan_execute.plan_and_execute import PlanAndExecute
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Bool, Int16
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_srvs.srv import Empty
import math
import copy
import time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import openai

class State(Enum):
    """
    Current state of the system.

    Determines what the main timer function should be doing on each iteration
    """
    IDLE = auto()
    APPROACHING_OBJECT = auto()
    GRASPING_OBJECT = auto()
    APPROACHING_DESTINATION = auto()
    PLACING_OBJECT = auto()
    RETURNING_TO_BASE = auto()

"""
class Targets(Enum):
    APPLE = "apple"
    BANANA = "banana"
    EGGPLANT = "eggplant"
    GREEN_BEANS = "green beans"
    TOP_LEFT_CORNER = "top left corner"
    TOP_RIGHT_CORNER = "top right corner"
    MIDDLE = "middle"
    BOTTOM_LEFT_CORNER = "bottom left corner"
    BOTTOM_RIGHT_CORNER = "bottom right corner"
"""    

class Pick_And_Place(Node):
    """
    TODO: write docstring
    """

    def __init__(self):
        super().__init__('pick_and_place')
        self.frequency = 100
        self.timer_period = 1 / self.frequency  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # initialize subscription to a custom message that contains the pose of a detected object
        self.detection_sub = self.create_subscription(DetectedObject, 'detected_object', self.detection_callback, 10)

        # create a dictionary with pick/place targets as keys and their corresponding poses as values
        self.targets = {
            "apple": Point(),
            "banana": Point(),
            "eggplant": Point(),
            "green beans": Point(),
            "top left corner": Point(x=0.5, y=0.5, z=0.0),
            "top right corner": Point(x=0.5, y=-0.5, z=0.0),
            "middle": Point(x=0.5, y=0.0, z=0.0),
            "bottom left corner": Point(x=0.5, y=0.5, z=0.0),
            "bottom right corner": Point(x=0.5, y=-0.5, z=0.0),
        }
    
        self.current_state = State.IDLE
        self.current_pick_target = None
        self.current_place_target = None

        self.plan_and_execute = PlanAndExecute(self)

    def timer_callback(self):
        """
        Main loop of the node. This function is called periodically at the frequency specified by self.timer_period.
        """
        if self.current_state == State.IDLE:
            return
        
    def detection_callback(self, msg):
        """
        Callback function for the detection subscription.
        """
        object_name = msg.object_name
        if object_name in self.targets.keys():
            self.targets[object_name] = msg.position

def pick_and_place_entry():
    rclpy.init()
    pick_and_place = Pick_And_Place()
    rclpy.spin(pick_and_place)
    pick_and_place.destroy_node()
    rclpy.shutdown()