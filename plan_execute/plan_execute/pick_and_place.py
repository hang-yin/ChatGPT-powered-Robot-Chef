import rclpy
from rclpy.node import Node
from enum import Enum, auto
from plan_execute_interface.srv import GoHere, Place
from plan_execute.plan_and_execute import PlanAndExecute
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Int16
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_srvs.srv import Empty
import math
import copy
import time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class Pick_And_Place(Node):
    """
    TODO: write docstring
    """

    def __init__(self):
        super().__init__('pick_and_place')

def pick_and_place_entry():
    rclpy.init()
    pick_and_place = Pick_And_Place()
    rclpy.spin(pick_and_place)
    pick_and_place.destroy_node()
    rclpy.shutdown()