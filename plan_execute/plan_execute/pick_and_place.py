import rclpy
from rclpy.node import Node
from enum import Enum, auto
from plan_execute_interface.srv import GoHere, Place, Instruction
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

GPT_CONTEXT = """
# move all fruits and vegetables to the top left corner.
robot.pick_and_place(apple, top left corner)
robot.pick_and_place(eggplant, top left corner)
robot.pick_and_place(banana, top left corner)
robot.pick_and_place(green beans, top left corner)
done()

# put the fruits in the top right corner.
robot.pick_and_place(apple, top right corner)
robot.pick_and_place(banana, top right corner)
done()

# move the vegetables to the middle.
robot.pick_and_place(eggplant, middle)
robot.pick_and_place(green beans, middle)
done()
"""

class Pick_And_Place(Node):
    """
    TODO: write docstring
    """

    def __init__(self):
        super().__init__('pick_and_place')
        self.frequency = 100
        self.timer_period = 1 / self.frequency  # seconds
        self.cbgroup = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=self.cbgroup)

        # initialize subscription to a custom message that contains the pose of a detected object
        self.detection_sub = self.create_subscription(DetectedObject, 'detected_object', self.detection_callback, 10)

        # initialize service
        self.instruction_service = self.create_service(Instruction, '/instruction', self.instruction_callback)

        # create a dictionary with pick/place targets as keys and their corresponding poses as values
        self.pick_targets = {
            "apple": Point(),
            "banana": Point(),
            "eggplant": Point(),
            "green beans": Point(),
        }

        self.place_targets = {
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

        self.curr_instruction = None
        self.gpt_context = GPT_CONTEXT

        self.steps = []
        self.curr_pick_target = None
        self.curr_place_target = None

        self.goal_pose = Pose()
        self.goal_pose.position.x = 0.0
        self.goal_pose.position.y = 0.0
        self.goal_pose.position.z = 0.0
        self.goal_pose.orientation.x = 0.0
        self.goal_pose.orientation.y = 0.0
        self.goal_pose.orientation.z = 0.0
        self.goal_pose.orientation.w = 0.1

    async def timer_callback(self):
        """
        Main loop of the node. This function is called periodically at the frequency specified by self.timer_period.
        """
        if self.current_state == State.IDLE:
            return
        elif self.current_state == State.INTERPRET_INSTRUCTION:
            termination_string = "done()"
            gpt3_prompt = self.gpt_context + "\n#" + self.curr_instruction + "\n"
            # self.get_logger().info("GPT3 prompt: " + gpt3_prompt)
            options = self.make_options(termination_string=termination_string)
            num_tasks = 0
            max_tasks = 5
            selected_task = ""
            steps_text = []
            engine = "text-ada-001"
            while not selected_task == termination_string:
                num_tasks += 1
                if num_tasks > max_tasks:
                    break
                llm_scores, _ = self.gpt3_scoring(gpt3_prompt, options, verbose=True, engine=engine, print_tokens=False)
                selected_task = max(llm_scores, key=llm_scores.get)
                steps_text.append(selected_task)
                self.get_logger().info("#" + str(num_tasks) + " selected task: " + selected_task)
                # print(num_tasks, "Selecting: ", selected_task)
                gpt3_prompt += selected_task + "\n"
            self.steps = steps_text
            self.get_logger().info("Done with instruction: " + self.curr_instruction)
            for i, step in enumerate(steps_text):
                if step == '' or step == termination_string:
                    break
                self.get_logger().info("Step " + str(i) + ": " + step)
            self.current_state = State.PICK_READY
        elif self.current_state == State.PICK_READY:
            self.curr_pick_target = self.steps[0].split()[0]
            self.curr_place_target = self.steps[0].split()[2]
            pick_ready = copy.deepcopy(self.goal_pose)
            pick_ready.position.x = self.pick_targets[self.curr_pick_target].x
            pick_ready.position.y = self.pick_targets[self.curr_pick_target].y
            self.future = await self.plan_and_execute.plan_to_cartisian_pose(None,
                                                                             pick_ready,
                                                                             1.0,
                                                                             True)
        elif self.current_state == State.PICK:
            pass
        elif self.current_state == State.GRASP:
            pass
        elif self.current_state == State.PICK_RETURN:
            pass
        elif self.current_state == State.PLACE_READY:
            pass
        elif self.current_state == State.PLACE:
            pass
        elif self.current_state == State.RELEASE:
            pass
        elif self.current_state == State.PLACE_RETURN:
            # TODO: go to place return position
            # remove the first step from the list of steps
            self.steps.pop(0)
            # if there are no more steps, we are done executing the instruction
            if len(self.steps) == 0:
                self.current_state = State.HOME
            else:
                self.current_state = State.PICK_READY
        elif self.current_state == State.HOME:
            # TODO: return to home position
            self.current_state = State.IDLE
    
    def make_options(self, options_in_api_form=True, termination_string="done()"):
        options = []
        for pick in self.pick_targets:
            for place in self.place_targets:
                if options_in_api_form:
                    option = "{} -> {}".format(pick, place)
                else:
                    option = "Pick the {} and place it on the {}.".format(pick, place)
                options.append(option)
        options.append(termination_string)
        self.get_logger().info("Considering: " + str(options) + " options")
        return options
    
    def gpt3_scoring(self, query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
        if limit_num_options:
            options = options[:limit_num_options]
        verbose and print("Scoring", len(options), "options")
        gpt3_prompt_options = [query + option for option in options]
        response = self.gpt3_call(engine=engine, 
                                  prompt=gpt3_prompt_options, 
                                  max_tokens=0,
                                  logprobs=1, 
                                  temperature=0,
                                  echo=True,)
        
        scores = {}
        for option, choice in zip(options, response["choices"]):
            tokens = choice["logprobs"]["tokens"]
            token_logprobs = choice["logprobs"]["token_logprobs"]

            total_logprob = 0
            for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
                if option_start is None and not token in option:
                    break
                if token == option_start:
                    break
                total_logprob += token_logprob
                scores[option] = total_logprob

        for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
            verbose and print(option[1], "\t", option[0])
            if i >= 10:
                break

        return scores, response
    
    def gpt3_call(self, engine="text-ada-001", prompt="", max_tokens=128, temperature=0, logprobs=1, echo=False):
        response = openai.Completion.create(engine=engine, 
                                            prompt=prompt, 
                                            max_tokens=max_tokens, 
                                            temperature=temperature,
                                            logprobs=logprobs,
                                            echo=echo)
        return response
        
    def detection_callback(self, msg):
        """
        Callback function for the detection subscription.
        """
        object_name = msg.object_name
        if object_name in self.pick_targets.keys():
            self.pick_targets[object_name] = msg.position
            self.get_logger().info('Received pose of ' + object_name + ' at ' + str(msg.position))
        elif object_name in self.place_targets.keys():
            self.place_targets[object_name] = msg.position
    
    def instruction_callback(self, request, response):
        """
        Callback function for the instruction service.
        """
        self.curr_instruction = request.instruction
        self.get_logger().info('Received instruction: ' + self.curr_instruction)
        self.current_state = State.INTERPRET_INSTRUCTION
        return response

def pick_and_place_entry():
    openai_api_key = ""
    openai.api_key = openai_api_key
    rclpy.init()
    pick_and_place = Pick_And_Place()
    rclpy.spin(pick_and_place)
    pick_and_place.destroy_node()
    rclpy.shutdown()