import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HandActionNode(Node):
    def __init__(self):
        super().__init__('hand_action_node')

def main(args=None):
    """Start and spin the node."""
    rclpy.init(args=args)
    node = HandActionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
