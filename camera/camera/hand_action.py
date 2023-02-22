import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from enum import Enum, auto

class State(Enum):
    """The current state of the scan."""
    IDLE = auto()
    SCANNING = auto()

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
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # create subscriber for a Bool from /start_action_scan topic
        self.start_action_scan_sub = self.create_subscription(Bool,
                                                              '/start_action_scan',
                                                              self.start_action_scan_callback,
                                                              10)

        # create publisher for a Bool to /hand_action topic
        # 0 indicates grabbing, 1 indicates cutting
        self.hand_action_pub = self.create_publisher(Bool, '/hand_action', 10)

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
