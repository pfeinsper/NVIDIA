# Control

## Road-Following Jetbot

In this project we are using the Jetbot Road-Following code as it already integrates the Jetbot camera with the motors, so the code interface helps choosing the parameters for the robot control.

From this code, Jetbot provides three parameters from the robot motors: 

- Speed Gain 

    This parameter controls the rotation of the motors simultaneously so that it changes the speed of the robot for its navigation.

- Steering Gain

    The steering gain has the capability of changing the proportion of rotation between one motor and the other, as it allows the robot to make turns more easily.

- Derivative Gain

    Finally we have the derivative gain, that identifies changes on the system and act to avoid deviations on the current system.

![Control Equation](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/control_equation.png?raw=true)

### Imagem dos ganhos no código:


