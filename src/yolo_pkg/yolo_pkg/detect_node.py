import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import cvzone

import os
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


classNames = ["pole", "person"]

### LOAD MODEL
    # Get the path to the package directory
package_name = 'yolo_pkg'
file_name = 'best.pt'
package_path = get_package_share_directory(package_name)
    # Construct the path to the file
file_path = os.path.join(package_path, file_name)
model = YOLO(file_path)

#Creat Node Subcribe Topic /camera/image_raw
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")
        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.callback, 10)
        self.bridge = CvBridge()

    def callback(self, msg):
        bgr_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        res = model(bgr_frame, stream=True)
        for r in res:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                curClass = classNames[cls]
                if curClass == "pole":
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cvzone.cornerRect(bgr_frame, (x1, y1, w, h))
                    cv2.circle(bgr_frame, (cx, cy), 5, (255, 0, 255), -1)

        cv2.imshow("Image_Raw", bgr_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
