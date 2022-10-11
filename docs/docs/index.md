# About

This is an open-source project of a MVP (Minimum Valiable Product) of a autonomous car using [Jetson Nano]("https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/") and integrating some Nvidia tools, such as [DeepStream SDK]("https://developer.nvidia.com/deepstream-sdk"), [Nvidia GPU Cloud]("https://www.nvidia.com/en-us/gpu-cloud/"), [TensorRT]("https://developer.nvidia.com/tensorrt"), [CUDA Toolkit]("https://developer.nvidia.com/cuda-toolkit") and [Triton]("https://developer.nvidia.com/nvidia-triton-inference-server").


## Pre-requisites

* Jetson Nano 4GB
* SDCard 64GB
* [Jetbot Toolkit]("https://jetbot.org/master/") 

!!! warning
    At first our recommendation is to follow the basic Jetbot setup from this website [https://jetbot.org/master/]("https://jetbot.org/master/"), since it shows how to build the robot, install the Jetpack OS on Jetson Nano and assemble the hardware and software of the robot.

    So just to make sure, the Jetbot tutorial parts you need to follow are [Bill of Materials]("https://jetbot.org/master/bill_of_materials.html"), [Hardware Setup]("https://jetbot.org/master/hardware_setup.html") and then for Software setup you can choose either to use [SD Card Image]("https://jetbot.org/master/software_setup/sd_card.html") or [Docker Container]("https://jetbot.org/master/software_setup/docker.html"). We recommend you to use the Docker Container setup, because the grapical interface takes a reasonable amount of CPU from the Jetson Nano.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

| First Header | Second Header | Third Header |
| ------------ | ------------- | ------------ |
| Content Cell | Content Cell  | Content Cell |
| Content Cell | Content Cell  | Content Cell |