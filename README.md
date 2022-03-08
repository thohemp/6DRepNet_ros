# **ROS: 6D Rotation Representation for Unconstrained Head Pose Estimation (Pytorch)**

Basic ROS implementation of 6DRepNet. For for information visit https://github.com/thohemp/6DRepNet.

<p align="center">
  <img src="https://github.com/thohemp/archive/blob/main/6DRepNet2.gif" alt="animated" />
</p>

# <div align="center"> **Quick Start**: </div>

```sh
git clone https://github.com/thohemp/6DRepNet_ros ros_workspace/src
catkin build 6DRepNet_ros
source devel/setup.bash # Source workspace
cd src/6DRepNet_ros
chmod +x 6drepnet_node.py
```
### Prerequisites:
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # Install required packages
pip install git+https://github.com/elliottzheng/face-detection.git@master # Face detector
```


##  **Camera Demo**:

```sh
roscore # start roscore
rosrun 6DRepNet_ros 6drepnet_node.py  --snapshot 6DRepNet_300W_LP_AFLW2000.pth \
                --cam 0
```


##  **Image topic Demo**:

```sh
roscore # start roscore
rosrun 6DRepNet_ros 6drepnet_node.py  --snapshot 6DRepNet_300W_LP_AFLW2000.pth \
                --image_topic /image/image_raw/compressed
```

### Topics:
* sixdrepnet/image

Show topic:
```
rosrun image_view image_view image:=/sixdrepnet/image
```


