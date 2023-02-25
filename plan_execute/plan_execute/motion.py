import rclpy
from rclpy.node import Node
from enum import Enum, auto
from plan_execute_interface.srv import GoHere, Place, Instruction
from plan_execute_interface.msg import DetectedObject
from plan_execute.plan_and_execute import PlanAndExecute
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Bool, Int16, Int64, String
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
    START = auto()
    IDLE = auto()
    INTERPRET_INSTRUCTION = auto()
    PICK_READY = auto()
    PICK = auto()
    GRASP = auto()
    PICK_RETURN = auto()
    PLACE_READY = auto()
    PLACE = auto()
    RELEASE = auto()
    PLACE_RETURN = auto()
    HOME = auto()
    PLACE_PLANE = auto()
    CARTESIAN = auto()
    HAND_DETECTION = auto()

class Motion(Node):
    """
    TODO: write docstring
    """

    def __init__(self):
        super().__init__('motion')
        self.frequency = 100
        self.timer_period = 1 / self.frequency  # seconds
        self.cbgroup = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=self.cbgroup)

        # initialize subscription to a custom message that contains the pose of a detected object
        self.detection_sub = self.create_subscription(DetectedObject, '/detected_object', self.detection_callback, 10)

        # initialize subscription to a string message that contains the instruction
        self.instruction_sub = self.create_subscription(String, '/gpt_instruction', self.instruction_callback, 10)

        # initialize publisher to publish a Bool to /start_action_scan topic
        self.start_action_scan_pub = self.create_publisher(Bool, '/start_action_scan', 10)

        # initialize subscription to a Int64 message that tells the current hand action
        self.hand_action_sub = self.create_subscription(Int64, '/hand_action', self.hand_action_callback, 10)

        # create a dictionary with pick/place targets as keys and their corresponding poses as values
        self.pick_targets = {
            "strawberry": Point(x=0.397, y=0.275, z=0.04),
            "banana": Point(x=0.465, y=0.0, z=0.030),
            "eggplant": Point(x=0.465, y=0.0, z=0.032),
            "green beans": Point(x=0.465, y=0.0, z=0.032),
        }

        self.pick_widths = {
            "apple": 0.06,
            "banana": 0.03,
            "eggplant": 0.08,
            "green beans": 0.07,
            "strawberry": 0.04,
        }

        self.place_targets = {
            "top left corner": Point(x=0.5, y=0.5, z=0.0),
            "top right corner": Point(x=0.5, y=-0.5, z=0.0),
            "middle": Point(x=0.5, y=0.0, z=0.0),
            "bottom left corner": Point(x=0.5, y=0.5, z=0.0),
            "bottom right corner": Point(x=0.5, y=-0.5, z=0.0),
            "basket": Point(x=0.3, y=-0.23, z=0.18),
        }
    
        self.current_state = State.START

        self.movegroup = None
        self.plan_and_execute = PlanAndExecute(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.curr_instruction = None

        self.steps = []
        self.curr_pick_target = None
        self.curr_place_target = None

        self.goal_pose = Pose()
        self.goal_pose.position.x = 0.0
        self.goal_pose.position.y = 0.0
        self.goal_pose.position.z = 0.0
        self.goal_pose.orientation.x = 1.0
        self.goal_pose.orientation.y = 0.0
        self.goal_pose.orientation.z = 0.0
        self.goal_pose.orientation.w = 0.0

        self.future = None
        self.ct = 0

        self.home_pose = Pose()
        self.home_pose.position.x = 0.3
        self.home_pose.position.y = 0.0
        self.home_pose.position.z = 0.487
        self.home_pose.orientation.x = 1.0
        self.home_pose.orientation.y = 0.0
        self.home_pose.orientation.z = 0.0
        self.home_pose.orientation.w = 0.0

        self.started_cutting = False
    
    def instruction_callback(self, msg):
        """
        Callback function for the instruction subscription.
        """
        self.curr_instruction = msg.data
        self.current_state = State.INTERPRET_INSTRUCTION
    
    def hand_action_callback(self, msg):
        """
        Callback function for the hand action subscription.
        """
        if msg.data == 0:
            if not self.started_cutting:
                return
            else:
                self.started_cutting = False
                self.current_state = State.PICK_READY
        elif msg.data == 1:
            self.started_cutting = True

    def cart_callback(self, request, response):
        """
        Call a custom service that takes one Pose of variable length, a regular Pose, and a bool.

        The user can pass a custom start postion to the service and a desired end goal. The boolean
        indicates whether to plan or execute the path.
        """
        self.goal_pose = request.goal_pose
        self.execute = True
        self.start_pose = None
        self.current_state = State.CARTESIAN
        response.success = True
        return response

    async def timer_callback(self):
        """
        Main loop of the node. This function is called periodically at the frequency specified by self.timer_period.
        """
        if self.current_state == State.START:
            if self.ct == 100:
                self.current_state = State.PLACE_PLANE
                self.ct = 0
            else:
                self.ct += 1
        elif self.current_state == State.PLACE_PLANE:
            self.current_state = State.IDLE
            await self.place_plane()
        elif self.current_state == State.IDLE:
            return
        elif self.current_state == State.INTERPRET_INSTRUCTION:
            termination_string = "done()"
            selected_task = ""
            # break current instruction into lines
            temp_steps = self.curr_instruction.splitlines()
            # get rid of empty lines and lines that don't start with "robot", "human", or "done()"
            self.steps = [step for step in temp_steps if step != ''
                                                         and (step.startswith("robot")
                                                         or step.startswith("human")
                                                         or step.startswith(termination_string))]
            for i, step in enumerate(self.steps):
                if step == '' or step == termination_string:
                    break
                self.get_logger().info("Step " + str(i) + ": " + step)
            self.current_state = State.PICK_READY
        elif self.current_state == State.PICK_READY:
            if len(self.steps) == 1:
                self.get_logger().info("Done executing instruction")
                self.steps.pop(0)
                self.current_state = State.HOME
                return
            # while first step starts with "human", we pop and go to HAND_DETECTION state
            if self.steps[0].startswith("human"):
                while self.steps[0].startswith("human"):
                    self.steps.pop(0)
                self.current_state = State.HAND_DETECTION
                return
            # self.curr_pick_target = self.steps[0].split()[0]
            split = self.steps[0][21:-1].split(', ')
            self.curr_pick_target = split[0]
            self.get_logger().info("Current pick target: " + self.curr_pick_target)
            # self.curr_place_target = self.steps[0].split()[2]
            self.curr_place_target = split[1]
            self.get_logger().info("Current place target: " + self.curr_place_target)
            pick_ready = copy.deepcopy(self.goal_pose)
            pick_ready.position.x = self.pick_targets[self.curr_pick_target].x
            pick_ready.position.y = self.pick_targets[self.curr_pick_target].y
            pick_ready.position.z = 0.18
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(start_pose=None,
                                                                             end_pose=pick_ready,
                                                                             v=0.5,
                                                                             execute=True)
            self.current_state = State.PICK
        elif self.current_state == State.PICK:
            pick = copy.deepcopy(self.goal_pose)
            pick.position.x = self.pick_targets[self.curr_pick_target].x
            pick.position.y = self.pick_targets[self.curr_pick_target].y
            # pick.position.z = self.pick_targets[self.curr_pick_target].z
            pick.position.z = 0.028
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(start_pose=None,
                                                                             end_pose=pick,
                                                                             v=0.5,
                                                                             execute=True)
            self.current_state = State.GRASP
        elif self.current_state == State.GRASP:
            self.future = await self.plan_and_execute.grab(self.pick_widths[self.curr_pick_target])
            time.sleep(4)
            self.current_state = State.PICK_RETURN
        elif self.current_state == State.PICK_RETURN:
            pick_return = copy.deepcopy(self.goal_pose)
            pick_return.position.x = self.pick_targets[self.curr_pick_target].x
            pick_return.position.y = self.pick_targets[self.curr_pick_target].y
            pick_return.position.z = 0.18
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(start_pose=None,
                                                                             end_pose=pick_return,
                                                                             v=0.5,
                                                                             execute=True)
            self.current_state = State.PLACE_READY
        elif self.current_state == State.PLACE_READY:
            place_ready = copy.deepcopy(self.goal_pose)
            place_ready.position.x = self.place_targets[self.curr_place_target].x
            place_ready.position.y = self.place_targets[self.curr_place_target].y
            place_ready.position.z = 0.18
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(start_pose=None,
                                                                             end_pose=place_ready,
                                                                             v=0.5,
                                                                             execute=True)
            self.current_state = State.PLACE
        elif self.current_state == State.PLACE:
            place = copy.deepcopy(self.goal_pose)
            place.position.x = self.place_targets[self.curr_place_target].x
            place.position.y = self.place_targets[self.curr_place_target].y
            # place.position.z = self.place_targets[self.curr_place_target].z
            place.position.z = 0.18
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(start_pose=None,
                                                                             end_pose=place,
                                                                             v=0.5,
                                                                             execute=True)
            self.current_state = State.RELEASE
        elif self.current_state == State.RELEASE:
            self.future = await self.plan_and_execute.release()
            time.sleep(4)
            self.current_state = State.PLACE_RETURN
        elif self.current_state == State.PLACE_RETURN:
            place_return = copy.deepcopy(self.goal_pose)
            place_return.position.x = self.place_targets[self.curr_place_target].x
            place_return.position.y = self.place_targets[self.curr_place_target].y
            place_return.position.z = 0.18
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(start_pose=None,
                                                                             end_pose=place_return,
                                                                             v=0.5,
                                                                             execute=True)
            # print all the steps in self.steps
            for i in range(len(self.steps)):
                self.get_logger().info("Step " + str(i) + ": " + self.steps[i])
            # remove the first step from the list of steps
            if len(self.steps) > 0:
                self.steps.pop(0)
                self.current_state = State.PICK_READY
            else:
                self.current_state = State.HOME
        elif self.current_state == State.HOME:
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(start_pose=None,
                                                                             end_pose=self.home_pose,
                                                                             v=0.5,
                                                                             execute=True)
            self.current_state = State.IDLE
        elif self.current_state == State.HAND_DETECTION:
            msg = Bool()
            msg.data = True
            self.hand_detection_pub.publish(msg)
            self.current_state = State.IDLE
    
    async def place_plane(self):
        """Places a plane for the table in RVIZ."""
        plane_pose = Pose()
        plane_pose.position.x = 0.0
        plane_pose.position.y = 0.0
        plane_pose.position.z = -0.14
        plane_pose.orientation.x = 0.0
        plane_pose.orientation.y = 0.0
        plane_pose.orientation.z = 0.0
        plane_pose.orientation.w = 1.0
        self.get_logger().info("Placing plane")
        await self.plan_and_execute.place_block(plane_pose, [10.0, 10.0, 0.1], 'plane')
        self.get_logger().info("Plane placed")
        
    def detection_callback(self, msg):
        """
        Callback function for the detection subscription.
        """
        object_name = msg.object_name
        # perform a static transform for the detected object
        # we want a transform from the camera frame to the hand tcp frame
        obj_pose = Point()
        x_offset = 0.3064 - 0.065 + 0.134
        y_offset = -0.04
        obj_pose.x = -msg.position.x + x_offset
        obj_pose.y = msg.position.y + y_offset
        obj_pose.z = msg.position.z
        if object_name in self.pick_targets.keys():
            self.pick_targets[object_name] = obj_pose
            self.get_logger().info('Received pose of ' + object_name + ' at ' + str(obj_pose))
        elif object_name in self.place_targets.keys():
            self.place_targets[object_name] = msg.position

def motion_entry():
    rclpy.init()
    motion_node = Motion()
    rclpy.spin(motion_node)
    motion_node.destroy_node()
    rclpy.shutdown()