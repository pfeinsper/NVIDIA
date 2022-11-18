<div align="center">
  <a src="https://www.nvidia.com/en-us/">
    <img src="https://github.com/pfeinsper/NVIDIA/blob/main/docs/docs/images/Vertical_Logo/NV_Logo_2D.png?raw=true" height="100"/>
  </a>
  <h1>Nvidia HPC Autopilot</h1>
</div>

This is an open-source project of a MVP (Minimum Valiable Product) of a autonomous car using [Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/) that respects brazilian traffic signs and integrates some Nvidia tools, such as [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), [Nvidia GPU Cloud](https://www.nvidia.com/en-us/gpu-cloud/), [TensorRT](https://developer.nvidia.com/tensorrt), [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), [Triton](https://developer.nvidia.com/nvidia-triton-inference-server) and [Jetbot](https://jetbot.org/master/) as the main tool for the project.

## Project Architecture

<div align="center">
  <img src="https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/architechture.png?raw=true"/>
</div>

The main workflow of the project shown on this figure above. The architecture of the project is based on four sections:

- Autopilot 
- Neural Networks
    - Detector
    - Classifier
- Broker MQTT 
- Controller

For more details on how to start with our project, check our documentation: 

[Docs](https://github.com/pfeinsper/NVIDIA)

So the robot starts following the road using computer vision and the controller parameters to stay on track, as the Jetbot sees any traffic sign, for instance a stop sign, the robot immediately detect the stop sign as a roadsign from the detector model and uses it to classify the roadsign as a stop sign by the classifier model. After this, the classifier connects with a broker that uses asyncronous message to communicate with the autopilot and tells the robot to stop, so the motors receive a speed gain of zero and the robot stops on the track, respecting the traffic sign identified.

<div align="center">
  <h3>Robot Image</h3>
  <a>
    <img src="https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/jetbot.png?raw=true"/>
  </a>
</div>