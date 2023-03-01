from flask import Flask, redirect, url_for, request, render_template, jsonify
# from flask_ask import Ask, statement, question, session
from chatgpt_wrapper import ChatGPT
import threading
import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

global gpt_response
gpt_response = None

pre_prompt = """Given the following context, what follows after this statement "# help me prepare """


old_post_prompt = """
"?  Follow the syntax of the below context and only output statements that either start with robot.pick_and_place or human.cut or done()

Context: 

available pick objects [apple, banana, eggplant, green beans, corn, carrot, kiwi, egg, strawberry, orange, yellow pepper, tomato sauce, cheese, milk]
available place_objects [soup pot, fry pan, pressure cooker, plate, electric stove, basket, sink, chopping board, blender]

# help me cook eggplant parmesan.
robot.pick_and_place(eggplant, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(eggplant)
robot.pick_and_place(eggplant, fry pan)
robot.pick_and_place(tomato_sauce, fry pan)
robot.pick_and_place(cheese, fry pan)
done()

# help me prepare a fruit salad.
robot.pick_and_place(apple, chopping board)
robot.pick_and_place(banana, chopping board)
robot.pick_and_place(kiwi, chopping board)
robot.pick_and_place(strawberry, chopping board)
robot.pick_and_place(orange, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(apple)
human.cut(banana)
human.cut(kiwi)
human.cut(strawberry)
human.cut(orange)
done()

# help me prepare tomato and vegetable omelette. 
robot.pick_and_place(tomato, chopping board)
robot.pick_and_place(eggplant, chopping board)
robot.pick_and_place(yellow_pepper, chopping board)
robot.pick_and_place(green_beans, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(tomato)
human.cut(eggplant)
human.cut(yellow_pepper)
human.cut(green_beans)
robot.pick_and_place(tomato, fry pan)
robot.pick_and_place(eggplant, fry pan)
robot.pick_and_place(yellow_pepper, fry pan)
robot.pick_and_place(green_beans, fry pan)
robot.pick_and_place(egg, fry pan)
done()

# help me prepare a banana and strawberry smoothie.
robot.pick_and_place(banana, chopping board)
robot.pick_and_place(strawberry, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(banana)
human.cut(strawberry)
robot.pick_and_place(banana, blender)
robot.pick_and_place(strawberry, blender)
robot.pick_and_place(milk, blender)
done()
"""

post_prompt = """
"?  Follow the syntax of the below context and only output statements that either start with robot.pick_and_place or human.cut or done()

Context: 

available pick objects [banana, eggplant, green beans, kiwi, egg, strawberry, orange, yellow pepper, tomato sauce, cheese, milk]
available place_objects [soup pot, fry pan, pressure cooker, plate, electric stove, chopping board, blender]

# help me cook eggplant parmesan.
robot.pick_and_place(eggplant, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(eggplant)
robot.pick_and_place(eggplant, fry pan)
robot.pick_and_place(tomato_sauce, fry pan)
robot.pick_and_place(cheese, fry pan)
done()

# help me prepare a fruit salad.
robot.pick_and_place(banana, chopping board)
robot.pick_and_place(kiwi, chopping board)
robot.pick_and_place(strawberry, chopping board)
robot.pick_and_place(orange, chopping board)
human.cut(banana)
human.cut(kiwi)
human.cut(strawberry)
human.cut(orange)
done()

# help me prepare tomato and vegetable omelette. 
robot.pick_and_place(tomato, chopping board)
robot.pick_and_place(eggplant, chopping board)
robot.pick_and_place(yellow_pepper, chopping board)
robot.pick_and_place(green_beans, chopping board)
human.cut(tomato)
human.cut(eggplant)
human.cut(yellow_pepper)
human.cut(green_beans)
robot.pick_and_place(tomato, fry pan)
robot.pick_and_place(eggplant, fry pan)
robot.pick_and_place(yellow_pepper, fry pan)
robot.pick_and_place(green_beans, fry pan)
robot.pick_and_place(egg, fry pan)
done()

# help me prepare a banana and strawberry smoothie.
robot.pick_and_place(banana, chopping board)
robot.pick_and_place(strawberry, chopping board)
human.cut(banana)
human.cut(strawberry)
robot.pick_and_place(banana, blender)
robot.pick_and_place(strawberry, blender)
robot.pick_and_place(milk, blender)
done()
"""

class FlaskNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.app = Flask(__name__)

        self.steps_ready = False

        # define Flask routes here
        @self.app.route('/')
        def homepage():
            return "Welcome, this is my super empty home page for my winter project!"

        @self.app.route('/alexa', methods=['POST'])
        def handle_request():
            json_data = request.get_json()
            food_string = json_data['string_key']
            if food_string != 'void':
                prompt = pre_prompt + food_string + post_prompt
                t = threading.Thread(target=self.gpt_call, args=(prompt,))
                t.start()
                """
                # Wait for global variable to be set
                global gpt_response
                while gpt_response is None:
                    time.sleep(0.1)
                """
                # gpt_response = gpt_call(prompt)
                # Process the JSON data as needed
                # ...
                # Send a response
                response = {'message': 'Processing request for ' + food_string + ' with ChatGPT, please wait for a moment and ask me for the steps!'}
                # gpt_response = None
                return jsonify(response)
            else:
                global gpt_response
                if gpt_response is None:
                    response = {"message": "Sorry, I'm still processing your request, please wait for a moment and ask me for the steps!"}
                    return jsonify(response)
                else:
                    response = {"message": gpt_response}
                    gpt_response = None
                    self.steps_ready = True
                    return jsonify(response)

        # start Flask app on a separate thread
        self.app_thread = threading.Thread(target=self.app.run)
        self.app_thread.start()

        # initialize a publisher to /gpt_instruction topic with String message type
        self.instruction_publisher_ = self.create_publisher(String, '/gpt_instruction', 10)

        # initialize a timer
        self.timer = self.create_timer(0.5, self.timer_callback)
    
    def gpt_call(self, prompt):
        bot = ChatGPT()
        response = bot.ask(prompt)
        # store the response in the global variable
        global gpt_response
        gpt_response = response
        # log the response
        self.get_logger().info('Response: %s' % response)
        gpt_msg = String()
        gpt_msg.data = response
        self.instruction_publisher_.publish(gpt_msg)

    def timer_callback(self):
        if self.steps_ready:
            msg = String()
            msg.data = gpt_response
            self.instruction_publisher_.publish(msg)
            self.steps_ready = False

    def __del__(self):
        # stop Flask app when node is destroyed
        self.app_thread.join()

def main(args=None):
    """Start and spin the node."""
    rclpy.init(args=args)
    node = FlaskNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
