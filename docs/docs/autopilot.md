# Autopilot

For the autopilot section of the project we built a line tracker algorithm using pure computer vision. This piece of code use OpenCV and it creates a mask to identify the yellow line of the road, as long as the yellow line is detected, we use a linear regression to preview all the points that the robot should follow and by using the angle from linear regression the robot calculates the best steering gain, speed gain and derivative gain using a control equation. Finally these informations are sent on a asynchronous message broker to the robot motors.

## Control parameters

From this code, Jetbot provides three parameters from the robot motors: 

- Speed Gain 

    This parameter controls the rotation of the motors simultaneously so that it changes the speed of the robot for its navigation.

- Steering Gain

    The steering gain has the capability of changing the proportion of rotation between one motor and the other, as it allows the robot to make turns more easily.

- Derivative Gain

    Finally we have the derivative gain, that identifies changes on the system and act to avoid deviations on the current system.

Using the computer vision algorithm with the asynchronous message broker the Jetbot interacts with the motors using the parameters sent by the algorithm. 

![Autopilot Algorithm](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/lane_detector.png?raw=true)

For this algorithm of the autopilot we have two examples, one using the Broker MQTT to comunicate with the robot and other that uses just the camera to identify the middle line from the road and preview the commands that the robot should follow, but it doesn`t comunicate with the robot motors.

You can find these algorithms on this links below:

- [Computer Vision using Broker MQTT](https://github.com/pfeinsper/NVIDIA/blob/main/mqtt/main.py)
- [Just the Computer Vision algorithm](https://github.com/pfeinsper/NVIDIA/blob/main/mqtt/lanedetector.py)

## Running the autopilot

In order to run this project section you just need to run the python code remembering to change the camera input number on the code and change the IP ADDRESS too, these steps are shown below:

Verifing if the camera input number is correct:

```python

broker = 'IP_ADDRESS'
.
.
.

class LaneDetector:
    def __init__(self):
        # self.video_path = video_path
        self.cap = cv2.VideoCapture("CAMERA_INPUT_NUMBER")
        self.frame = None
        self.mask = None
        self.contours = None
        self.m = None
        self.p1 = None
        self.p2 = None
        self.point_center = None 
    .
    .
    .
```

Running the code:

```bash
#running with mqtt
python3 main.py

#running without mqtt
python3 lanedetector.py
```