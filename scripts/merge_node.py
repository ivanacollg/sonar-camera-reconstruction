#!/usr/bin/env python3

# Python libraries
import numpy as np
import threading

# OpenCV
import cv2
import cv_bridge

# ROS libraries
import rospy

# ROS Messages
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

#Custom ROS Messages
from sonar_oculus.msg import OculusPing

# Custom libraries
from merge import MergeFunctions

class MergeNode:
    """
    This ROS node merges sonar, camera, and odometry data to create a fused point cloud 
    and segmented images for enhanced underwater perception.
    """
    def __init__(self, ns="~"):
        """
        Initializes the MergeNode class, setting up ROS publishers, subscribers, and processing parameters.

        Parameters:
        - ns (str): Namespace for retrieving ROS parameters. Default is "~" (private namespace).
        """
        rospy.init_node('merge_node', anonymous=True)
        self.publish_rate = rospy.get_param("~publish_rate", 5)

        # Subscribers
        self.sonar_sub = rospy.Subscriber(rospy.get_param(ns + "sonar_sub"), OculusPing, callback=self.sonar_callback, queue_size=1)
        self.odom_sub  = rospy.Subscriber(rospy.get_param(ns + "odom_sub"), Odometry, callback=self.odom_callback, queue_size=1)
        self.image_sub = rospy.Subscriber(rospy.get_param(ns + "image_sub"), CompressedImage, self.image_callback,  queue_size = 1)
        # Publishers
        self.segmented_image_pub = rospy.Publisher(rospy.get_param(ns + "segmented_image_pub"), CompressedImage, queue_size=1)
        self.merge_cloud_pub = rospy.Publisher(rospy.get_param(ns + "merge_cloud_pub"), PointCloud2, queue_size=1)
        self.feature_image_pub = rospy.Publisher(rospy.get_param(ns + "feature_image_pub"), CompressedImage, queue_size=1)
        
        # Sonar to Camera Transfromation Matrix
        Ts_c = rospy.get_param(ns + "Ts_c")
        Ts_c = np.array(Ts_c).reshape((4, 4))
        # Get other merge parameters
        min_pix = rospy.get_param(ns + "min_area")
        threshold_inv = rospy.get_param(ns + "threshold_inv")
        self.merge = MergeFunctions(Ts_c, min_pix, threshold_inv)

        # Get monocular camera parameters
        rgb_width = rospy.get_param(ns + "image_width")
        rgb_height = rospy.get_param(ns + "image_height")
        K = rospy.get_param(ns + "camera_matrix/data")
        D = rospy.get_param(ns + "distortion_coefficients/data")
        K = np.array(K).reshape((3, 3))
        D = np.array(D)
        self.merge.set_camera_params(K, D, rgb_width, rgb_height)
       
        # Sonar Prameters
        sonar_range = rospy.get_param(ns + "sonarRange") # default value, reads in new value from msg
        detector_threshold = rospy.get_param(ns + "thresholdHorizontal")
        vertical_FOV = rospy.get_param(ns + "verticalAperture")
        self.merge.set_sonar_params(sonar_range, detector_threshold, vertical_FOV)
        #read in CFAR parameters
        Ntc = rospy.get_param(ns + "CFAR/Ntc")
        Ngc = rospy.get_param(ns + "CFAR/Ngc")
        Pfa = rospy.get_param(ns + "CFAR/Pfa")
        rank = rospy.get_param(ns + "CFAR/rank")
        # define the CFAR detector
        self.merge.init_CFAR(Ntc, Ngc, Pfa, rank)

        # define laser fields for fused point cloud
        self.laserFields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # the threading lock
        self.lock = threading.RLock()
        # CV bridge
        self.bridge_instance = cv_bridge.CvBridge()

        # Initialize sensor information
        self.image = None
        self.pose = None
        
        
    def sonar_callback(self, msg:OculusPing)->None:
        self.lock.acquire()
        self.merge.set_sensor_info(self.image, self.pose, msg)
        self.lock.release()

    def odom_callback(self, msg:Odometry)->None:
        self.pose = msg.pose.pose
    
    def image_callback(self, msg):
        #decode the compressed image
        self.image = self.bridge_instance.compressed_imgmsg_to_cv2(msg, "bgr8")
 
    def run(self):
        """
            Main loop for processing and publishing merged data.
        """
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            # Run merging algorithm
            point_cloud, segmented_image, stamp, feature_image = self.merge.merge_data()
            
            if point_cloud.size > 0:
                # Package the point cloud
                header = Header()
                header.frame_id = "map"
                header.stamp = stamp
                cloud_msg = pc2.create_cloud(header, self.laserFields, point_cloud)
                self.merge_cloud_pub.publish(cloud_msg)

            if segmented_image.size > 0:
                # Package the img
                image_msg = CompressedImage()
                image_msg.header.stamp = stamp
                image_msg.format = "jpeg"
                image_msg.data = np.array(cv2.imencode('.jpg',segmented_image)[1]).tobytes()
                self.segmented_image_pub.publish(image_msg)
                # Package the img
                image_msg.data = np.array(cv2.imencode('.jpg',feature_image)[1]).tobytes()
                self.feature_image_pub.publish(image_msg)



            rate.sleep()

if __name__ == "__main__":
    try:
        node = MergeNode()
        rospy.loginfo("Start sonar camera reconstruction node...")
        node.run()
    except rospy.ROSInterruptException:
        rospy.logwarn("ROS Interrupt Exception. Shutting down merge_node.")