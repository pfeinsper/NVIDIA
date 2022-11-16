# Running the project

This page section just shows how to run the project the fastest way possible on your JetBot.

At first you need to follow the setup from the Getting Started page, since you`ve installed the JetPack OS 4.6.1, JetBot 0.4.3 and all the JetBot construction, the next step is to clone the our Github repository:

```bash
git clone https://github.com/pfeinsper/NVIDIA
```

```
cd NVIDIA/
```

Then you need to clone the camera image source to other two sources, using both `/dev/video0`, `/dev/video1` and `/dev/video2`. For this step you need to install `ffmpeg` dependency.

```bash
xhost +
```

```bash
sudo modprobe v4l2loopback devices=2
```

```bash
ffmpeg -f video4linux2 -i /dev/video0 -codec copy -f v4l2 /dev/video1 -codec copy -f v4l2 /dev/video2
```

```bash
cd mqtt-example/
```

```bash
make broker
```

It is important to change the IP ADDRESS from the notebook file on `jetbot` folder located on the NVIDIA project

On Github you can see it on this link:

[https://github.com/pfeinsper/NVIDIA/blob/main/jetbot/fsm/mqtt.ipynb]("https://github.com/pfeinsper/NVIDIA/blob/main/jetbot/fsm/mqtt.ipynb")

```bash
cd jetbot/fsm

#open jupyter notebook
jupyter notebook
```

Access the `mqtt.ipynb` and change the broker variable that identifies the IP ADDRESS to your current IP ADDRESS

!!! Warning

    Since you have done that, don't run this notebook, because this is the last thing we need run.

To execute the computer vision algorithm you just need to run the `lanedetector.py` or the `main.py`. The `lanedetector.py` is used more for test cases.

```bash
python3 main.py

#or

python3 lanedetector.py
```

## Using docker container

In this part of the tutorial you will need to install docker ad nvidia-docker, for this you can use the reference below:

[https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html]("https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html")

The docker image that you will need for the project is located on DockerHub on this link below:

[https://hub.docker.com/r/edgardaon/jetson-deepstream-6.0-triton]("https://hub.docker.com/r/edgardaon/jetson-deepstream-6.0-triton")

```bash
# docker run --gpus -it --rm --net=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -w --privileged IMAGEM
```
!!! note
    If you had quit from the docker image, you can restart it and attach to restore all you have already done, using the commands below:

    ```bash
    #start the container
    docker start CONTAINER_ID

    #attach the container
    docker attach CONTAINER_ID
    ```

After you run the docker container you need to clone the project repository again:

```bash
git clone https://github.com/pfeinsper/NVIDIA
```

Then you need to change all the IP ADDRESSES to your current IP ADDRESS from the python files (`subscriber.py`, `main.py` and `lanedetector.py`) from the `mqtt-example` folder. 

```bash
cd /opt/nvidia/deepstream/deepstream/mqtt-example

#on this directory you can change the IP ADDRESSES from the variable "broker"
```

Now you will need to run the DeepStream python code that executes the traffic sign model based on the detectnet_v2, for this you can download the files below:

| Location            | Link                                |
| --------------------|-------------------------------------|
| Google Drive        |  [https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J")|

After this you can move these files to their directories from the docker container using the command below:

```bash
#moving the custom model files
mv calibration.bin resnet18_int8_detector.engine resnet18_detector.etlt labels.txt labels.txt /opt/nvidia/deepstream/deepstream/samples/models

#moving the deepstream config file
mv config_primary.txt /opt/nvidia/deepstream/deepstream/sources/apps/deepstream_python_apps/apps/deepstream-pfe
```

So after all these steps we have all setup and we just need to execute and integrate everything. At first we need to make sure the camera sources `/dev/video0`, `/dev/video1` and `/dev/video2` are listed on the docker container:

```bash
#here you may see all the video sources
ls /dev/video*
```

Next step is to execute another terminal with the same docker container using docker exec:

```
docker exec -ti CONTAINER_ID /bin/bash
```

So just to be clear, now we have two terminals running the docker container, one running the broker MQTT, one cloning camera sources. So then the last step is to run the python scripts for the computer vision and the Deepstream algorithm and finally run the jupyter notebook cell that uses the finite state machine to interact with the robot.

Running the python files:

On one docker container terminal you need to execute the command below:

```bash
cd /opt/nvidia/deepstream/deepstream/sources/apps/deepstream_python_apps/apps/deepstream-pfe

#this script runs on the video1 source
python3 deepstream-test-1-usb.py /dev/video1
```

On the second docker container you need to run these commands:

```bash
cd /opt/nvidia/deepstream/deepstream/mqtt-example

#in this script you need to make sure that the variable self.cap = cv2.VideoCapture(2) on the python file
python3 main.py
```

Finally outside de docker container (On the Jetson device), you can run the `mqtt.ipynb` notebook accessing the jupyter notebook from the JetBot container using the browser and just typing the CURRENT_IP_ADDRESS:8888/ and running the first cell that executes the robot.