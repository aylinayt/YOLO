import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Bool
import os


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node') # Initialize the node with a name
        self.subscription = self.create_subscription(Image, '/carla/ego_vehicle/rgb_front/image', self.image_callback, 10) # Queue size
        self.stop_publisher = self.create_publisher(Bool, '/carla/ego_vehicle/stop_flag', 10) # Queue size
        
        
        self.bridge = CvBridge() # Bridge to convert between ROS and OpenCV images
        self.class_names = open("coco.names").read().strip().split("\n") # Load class names
        self.net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights") # Load YOLOv3 model
        
        self.layer_names = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()] # Get the output layer names
        
        self.save_dir = "saved_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.image_count = 0 # To keep track of the number of saved images



    def image_callback(self, msg):
        boxes = []
        confidences = []
        class_ids = []
        # Convert the incoming ROS image message to an OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        (H, W) = image.shape[:2] # Get image dimensions (height, width
        
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        
        self.net.setInput(blob) # Set the blob as input to the YOLO network
        layer_outputs = self.net.forward(self.layer_names) # Perform forward pass to get detections
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:] # The confidence scores for each class
                class_id = np.argmax(scores) # Get the class ID with the highest confidence
                confidence = scores[class_id] # Get the highest confidence score
                
                if class_id == 0 and confidence > 0.25:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    # Save the detection results
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
               
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
               
                if len(indices) > 0:
                    stop_msg = Bool()
                    stop_msg.data = True # Set stop flag to True
                    self.stop_publisher.publish(stop_msg) # Publish the stop flag
                    self.get_logger().info('Stop sign detected! Stop flag published.') # Log the


        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = (0, 255, 0) # Bounding box color
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2) # Draw rectangle
                label = "{}: {:.2f}".format(self.class_names[class_ids[i]], confidences[i])
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                2) # Add labe
                image_filename = os.path.join(self.save_dir,
                f"image_{self.image_count:05d}.jpg")
                cv2.imwrite(image_filename, image)
                self.get_logger().info(f"Image saved as {image_filename}")
                self.image_count += 1
                # Publish the stop flag



def main(args=None):
    rclpy.init(args=args) # Initialize the ROS2 Python library
    node = YoloDetectionNode() # Create an instance of the YoloDetectionNode
    rclpy.spin(node) # Keep the node running
    node.destroy_node() # Destroy the node when shutting down
    rclpy.shutdown() # Shutdown the ROS2 Python library

if __name__ == '__main__':
    main() # Execute the main function when the script is run
