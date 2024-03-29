#!/usr/bin/env python3

from model import SixDRepNet, sixdrepnet_mobile_small
import math
import sys
import os
import argparse
import rospy
import rospkg

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import utils
from PIL import Image

from sensor_msgs.msg import Image as Image_msg
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header

import time
from face_detection import RetinaFace


class SixDRepNet_Node:
    def __init__(self):
        args = self.parse_args()
        cudnn.enabled = True
        self.gpu = args.gpu_id
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cam = args.cam_id
        self.snapshot_path = args.snapshot
        self.sub_topic = args.image_topic
        self.cpu = args.cpu
        self.c = args.c
        if not args.cpu: self.detector = RetinaFace(gpu_id=self.gpu) 
        else: self.detector = RetinaFace(-1)


        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        self.pub = rospy.Publisher('sixdrepnet/image', Image_msg,queue_size=10)
        self.pub_compr = rospy.Publisher('sixdrepnet/image/compressed', CompressedImage, queue_size=10)
        if self.sub_topic is not '' and self.c: self.sub = rospy.Subscriber(self.sub_topic, CompressedImage, self.image_callback) 
        elif self.sub_topic is not '' and not self.c: self.sub = rospy.Subscriber(self.sub_topic, Image_msg, self.image_callback_raw)
        #self.model = SixDRepNet(backbone_name='RepVGG-B1g2',
        #                    backbone_file='',
        #                    deploy=True,
        #                    pretrained=False)
        self.model = sixdrepnet_mobile_small()

        self.transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        print('Loading data.')
        # Load snapshot
        saved_state_dict = torch.load(os.path.join(
            self.snapshot_path))

        if 'model_state_dict' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state_dict)    
        if not args.cpu: self.model.to(self.dev)

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        if self.sub_topic == '':
            try:
                self.run_camera()
            except rospy.ROSInterruptException:
                pass
        else:
            print(self.sub_topic)
            rospy.spin()

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(
            description='Head pose estimation using the Hopenet network.')
        parser.add_argument('--gpu',
                            dest='gpu_id', help='GPU device id to use [0]',
                            default=0, type=int)
        parser.add_argument('--cam',
                            dest='cam_id', help='Camera device id to use [0]',
                            default=0, type=int)
        parser.add_argument('--snapshot',
                            dest='snapshot', help='Name of model snapshot.',
                            default='model/6drepnet_mobile_small.tar', type=str)
        parser.add_argument('--image_topic',
                            dest='image_topic', help='Compressed image topic to subscribe to.',
                            default='', type=str)
        parser.add_argument('--cpu', action='store_true', default=False)
        parser.add_argument('--c', action='store_true', default=False, help='Set if image_topic is compressed')

        args = parser.parse_args()
        return args

    def image_callback(self, data):
        #img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

        img = np.fromstring(data.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        #cv2.imshow("test", img)
        #(cv2.waitKey(5)
        start = time.time()
        self.inference(img)
        end = time.time()
        print('Head pose estimation: %2f ms'% ((end - start)*1000.))


    def image_callback_raw(self, data):
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.inference(img)


    def publish_image(self, imgdata):
        h, w, c = imgdata.shape
        image_temp=Image_msg()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'map'
        image_temp.height=h
        image_temp.width=w
        image_temp.encoding='bgr8'
        image_temp.data=np.array(imgdata).tostring()
        image_temp.header=header
        image_temp.step=w*3
        self.pub.publish(image_temp)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', imgdata)[1]).tostring()
        # Publish new image
        self.pub_compr.publish(msg)

    def run_camera(self):
        cap = cv2.VideoCapture(self.cam)
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while not rospy.is_shutdown():
            ret, frame = cap.read()

            self.inference(frame)

    def inference(self, frame):
        with torch.no_grad():

            faces = self.detector(frame)
              
            for box, landmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])         
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0,x_min-int(0.2*bbox_height))
                y_min = max(0,y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = self.transformations(img)
                img = torch.Tensor(img[None, :])
                if not self.cpu: img.to(self.dev)


                c = cv2.waitKey(1)
                if c == 27:
                    break
                    
                start = time.time()
            
                img = img.cuda()
                            
                R_pred = self.model(img)
                end = time.time()
             #  print('Head pose estimation: %2f ms'% ((end - start)*1000.))

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                
                #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(x_max-x_min)), y_min + int(.5*(y_max-y_min)), size = bbox_width)
                
            
            #cv2.imshow("Demo", frame)
            #cv2.waitKey(5)

            # Publish image 
            self.publish_image(frame)
            # self.loop_rate.sleep()

  

if __name__ == '__main__':
    rospy.init_node('sixdrepnet_head_pose', anonymous=True)
    sixdrepnet_node = SixDRepNet_Node()



    
