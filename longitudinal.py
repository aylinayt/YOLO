import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive # Message type for vehicle control
from std_msgs.msg import Bool # Message type for stop flag

class VehicleControlNode(Node):
    def __init__(self):
        super().__init__('vehicle_control_node') # Initialize the node with a name
        
        self.drive_publisher = self.create_publisher(
        AckermannDrive, # Message type for vehicle control
        '/carla/ego_vehicle/ackermann_cmd', 10) # Queue siz
        self.subscription = self.create_subscription(Bool, # Message type for stop flag
        '/carla/ego_vehicle/stop_flag', # Topic name for stop flag
        self.stop_callback, # Callback function when stop flag is
        10) # Queue size
        self.stop_flag = False
        self.timer = self.create_timer(0.1, self.drive_straight)


    def drive_straight(self):
        """Drive the vehicle straight unless stop flag is set."""
        if not self.stop_flag:
            drive_msg = AckermannDrive() # Create a new drive command message
            drive_msg.speed = 50.0 # Set the vehicle speed (in m/s)
            drive_msg.steering_angle = 0.0 # Set steering to drive straight
            self.drive_publisher.publish(drive_msg) # Publish the drive command
        else:
            self.brake() # Call the brake function if stop flag is set


    def stop_callback(self, msg):
        """Callback function to handle stop flag messages."""
        if msg.data:
            self.stop_flag = True # Set stop flag to True if stop signal is received


    def brake(self):
        """Stop the vehicle by setting the speed to zero."""
        drive_msg = AckermannDrive() # Create a new drive command message
        drive_msg.speed = 0.0 # Set the vehicle speed to zero
        self.drive_publisher.publish(drive_msg) # Publish the drive command
        self.get_logger().info('Braking...') # Log that the vehicle is braking

def main(args=None):
    rclpy.init(args=args) # Initialize the ROS2 Python library
    node = VehicleControlNode() # Create an instance of the VehicleControlNode
    rclpy.spin(node) # Keep the node running and processing callbacks
    node.destroy_node() # Clean up and destroy the node when shutting down
    rclpy.shutdown() # Shutdown the ROS2 Python library

if __name__ == '__main__':
    main() # Execute the main function when the script is run
