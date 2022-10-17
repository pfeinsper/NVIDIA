# Getting Started

![Jetbot](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/jetbot.png?raw=true){ align=left }

This is an open-source project of a MVP (Minimum Valiable Product) of a autonomous car using [Jetson Nano]("https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/") that respects brazilian traffic signs and integrates some Nvidia tools, such as [DeepStream SDK]("https://developer.nvidia.com/deepstream-sdk"), [Nvidia GPU Cloud]("https://www.nvidia.com/en-us/gpu-cloud/"), [TensorRT]("https://developer.nvidia.com/tensorrt"), [CUDA Toolkit]("https://developer.nvidia.com/cuda-toolkit"), [Triton]("https://developer.nvidia.com/nvidia-triton-inference-server") and [Jetbot]("https://jetbot.org/master/") as the main tool for the project. 

The main focus is to share this project with the community of developers so that can it be used as a starter for a autonomous vehicle or other robotics and computer vision projects.

## Pre-requisites

* Jetson Nano 4GB
* SDCard 64GB
* Jetbot Toolkit

!!! warning "Important"
    At first our recommendation is to follow the basic Jetbot setup from this website [https://jetbot.org/master/]("https://jetbot.org/master/"), since it shows how to build the robot, install the Jetpack OS on Jetson Nano and assemble the hardware and software of the robot.

    So just to make sure, the Jetbot tutorial parts you need to follow are [Bill of Materials]("https://jetbot.org/master/bill_of_materials.html"), [Hardware Setup]("https://jetbot.org/master/hardware_setup.html") and then for Software setup you can choose either to use [SD Card Image]("https://jetbot.org/master/software_setup/sd_card.html") or [Docker Container]("https://jetbot.org/master/software_setup/docker.html"). We recommend you to use the Docker Container setup, because the grapical interface takes a reasonable amount of CPU from the Jetson Nano.

## Project Architecture

![Architecture](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/architechture.png?raw=true)

The main workflow of the project shown on this figure above. The architecture of the project is based on four sections:

- Autopilot 
- Neural Networks
    - Detector
    - Classifier
- Broker MQTT 
- Controller

So the robot starts following the road using computer vision and the controller parameters to stay on track, as the Jetbot sees any traffic sign, for instance a stop sign, the robot immediately detect the stop sign as a roadsign from the detector model and uses it to classify the roadsign as a stop sign by the classifier model. After this, the classifier connects with a broker that uses asyncronous message to communicate with the autopilot and tells the robot to stop, so the motors receive a speed gain of zero and the robot stops on the track, respecting the traffic sign identified.
