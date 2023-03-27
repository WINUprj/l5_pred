#!/usr/bin/env python3
import rospy
import rospkg

import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Int32
from img_msgs.srv import DigitImage, DigitImageResponse
from sensor_msgs.msg import CompressedImage


DEBUG = True


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Two layers of CNN and single dense layer
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.mx_pool = nn.MaxPool2d(3)
        self.dp1 = nn.Dropout(0.3)
        self.dp2 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(12544, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size = x.shape[0]

        x = F.relu(self.conv1(x))
        x = self.bnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.mx_pool(x)

        x = self.dp1(x)
        x = x.view(batch_size, 1, -1)
        x = F.relu(self.fc1(x))
        # x = self.bnorm3(x)

        x = self.dp2(x)
        x = self.fc2(x)

        return x


def get_test_transform():
    return transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((42, 42)),
                                transforms.Normalize(mean = [0.5], std = [0.1])])


class DigitRecognitionNode(DTROS):
    def __init__(self, node_name="digit_recognition_node"):
        super(DigitRecognitionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        ) 
        
        # Get parameters
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Prepare model and related parameters/methods
        self.model = CNNModel()
        self.device = torch.device("cpu")
        self.test_transform = get_test_transform()
        
        # Prepare rospack for reading weights
        self.rospack = rospkg.RosPack()
        self.path = self.rospack.get_path("digit_recognition")
        self.weights_path = self.path + "/weights/"

        # self.sub_img = rospy.Subscriber(
        #     f"/{self.veh}/apriltag_detection_node/digit_img/compressed",
        #     CompressedImage,
        #     self.cb_img
        # )

        # self.pub_pred = rospy.Publisher(
        #     f"/local/{self.node_name}/pred",
        #     Int32,
        #     queue_size=1
        # )

        # Server
        self.srv_ints = rospy.Service("/local/digit_class", DigitImage, self.pred)
        self.srv_shutdown = rospy.Service("/local/shut_down", DigitImage, self.shutdown)

        rospy.on_shutdown(self.hook)

    def pred(self, req):
        # Preprocess the given image
        img = np.frombuffer(req.data.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = self.test_transform(img)
        img = img.unsqueeze(0)

        res = np.zeros(10)
        with torch.no_grad():
            for i in range(1, 6):
                w = f"cnn_model_{i}.pth"
                # Predict with different model weights
                self.model.load_state_dict(torch.load(self.weights_path + w,
                                           self.device))
                self.model.eval()
                y_hat = self.model(img.float()).squeeze(0)
                y_hat = F.softmax(y_hat, dim=1).numpy()
                res += y_hat[0]
            
            res /= 5

        if DEBUG:
            rospy.loginfo(f"Prediction: {np.argmax(res)}")

        return DigitImageResponse(int(np.argmax(res)))
    
    def shutdown(self, req):
        rospy.signal_shutdown("Shutting down digit recognition node")
        return DigitImageResponse(0)

    def hook(self):
        rospy.loginfo(f"Shutting down {self.node_name} node")
    
if __name__ == "__main__":
    node = DigitRecognitionNode()
    rospy.spin()
