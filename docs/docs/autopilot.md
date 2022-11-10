# Autopilot

For the autopilot section of the project we built a line tracker algorithm using pure computer vision. This piece of code use OpenCV and it creates a mask to identify the yellow line of the road, as long as the yellow line is detected, we use a linear regression to preview all the points that the robot should follow and by using the angle from linear regression the robot calculates the best steering gain, speed gain and derivative gain using a control equation. Finally these informations are sent on a asynchronous message broker to the robot motors.

Using the computer vision algorithm with the asynchronous message broker the Jetbot interacts with the motors using the parameters sent by the algorithm. 

# Control parameters 

- Speed Gain 

    This parameter controls the rotation of the motors simultaneously so that it changes the speed of the robot for its navigation.

- Steering Gain

    The steering gain has the capability of changing the proportion of rotation between one motor and the other, as it allows the robot to make turns more easily.

- Derivative Gain

    Finally we have the derivative gain, that identifies changes on the system and act to avoid deviations on the current system.

![Control Equation](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/control_equation.png?raw=true)

### Imagem dos ganhos no código:
        
