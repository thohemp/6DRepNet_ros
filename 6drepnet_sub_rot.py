#!/usr/bin/env python3

import sys
import os
import argparse
import rospy
import rospkg
from geometry_msgs.msg import Quaternion, PoseStamped, Pose
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo, Image

import numpy as np, cv2
import math

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians



class SixDRepNet_Processor_Node:

    def __init__(self):
        args = self.parse_args()
        self.cam = args.cam_id
        self.sub_topic = args.image_topic
        self.c = args.c

        self.pub_img_compr =  rospy.Publisher('sixdrepnet/process/image_raw/compressed', CompressedImage, queue_size=10)
        self.pub_img = rospy.Publisher('sixdrepnet/processed/image_raw/', Image, queue_size=10)

        rot_sub = message_filters.Subscriber('sixdrepnet/rotation', PoseStamped)
        image_sub = message_filters.Subscriber('/head_front_camera/color/image_raw/compressed', CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, rot_sub], 10,10)
        ts.registerCallback(self.rot_callback)

        rospy.spin()


    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(
            description='6DRepNet ROS topic processor.')
        parser.add_argument('--cam',
                            dest='cam_id', help='Camera device id to use [0]',
                            default=0, type=int)
        parser.add_argument('--image_topic',
                            dest='image_topic', help='Compressed image topic to subscribe to.',
                            default='/sixdrepnet/processed_image_raw', type=str)
        parser.add_argument('--cpu', action='store_true', default=False)
        parser.add_argument('--c', action='store_true', default=False, help='Set if image_topic is compressed')

        args = parser.parse_args()
        return args

    def rot_callback(self, img_data, rot_data):
        #print("{} {} ".format(img_data.header.stamp, rot_data.header.stamp))
        # Image
        img = np.fromstring(img_data.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        #img = np.frombuffer(img_data.data, dtype=np.uint8).reshape(img_data.height, img_data.width, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Rotation
        pose = rot_data.pose
        rot = pose.orientation
        roll, pitch, yaw = euler_from_quaternion(rot.x, rot.y, rot.z, rot.w)

        draw_axis(img, yaw*180/np.pi, pitch*180/np.pi, roll*180/np.pi, 50, 30, size=100)

        self.publish_image_compressed(img)
        self.publish_image(img)


    def publish_image(self, imgdata):
        h, w, c = imgdata.shape
        image_temp=Image()
        header = rospy.Header(stamp=rospy.Time.now())
        header.frame_id = 'map'
        image_temp.height=h
        image_temp.width=w
        image_temp.encoding='bgr8'
        image_temp.data=np.array(imgdata).tostring()
        image_temp.header=header
        image_temp.step=w*3
        self.pub_img.publish(image_temp)

    def publish_image_compressed(self, imgdata):
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        imgdata = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)

        msg.data = np.array(cv2.imencode('.jpg', imgdata)[1]).tostring()
        # Publish new image
        self.pub_img_compr.publish(msg)

if __name__ == '__main__':
    rospy.init_node('sixdrepnet_processor', anonymous=True)
    sixdrepnet_node = SixDRepNet_Processor_Node()
