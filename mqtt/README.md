# INSTRUCTIONS

1) Allow v4l2 to clone /dev/video0 to 2 devices
    ```console
    xhost +
    ```

    ```console
    sudo modprobe v4l2loopback devices=2
    ```

2) Use ffmpeg to clone /dev/video0 to /dev/video1 and /dev/video2
    ```console
    ffmpeg -f video4linux2 -i /dev/video0 -codec copy -f v4l2 /dev/video1 -codec copy -f v4l2 /dev/video2
    ```

3) Open another terminal (Ctrl + Shift + T) to make broker 
    ```console
    cd pfe/mqtt/
    ```

    ```console
    make broker
    ```

1) Open another terminal (Ctrl + Shift + T) to start and attach docker container (CONTAINER_ID = 45ce850560c3)
    ```console
    sudo docker start 45ce850560c3
    ```

    ```console
    sudo docker attach 45ce850560c3
    ```

### Using docker container

5) In this docker container, run the DeepStream demo test with usb cam
    ```console
    cd /opt/nvidia/deepstream/deepstream/sources/apps/deepstream_python_apps/apps/deepstream-test1-usbcam/
    ```

    ```console
    python3 deepstream_test_1_usb.py /dev/video0
    ```

6) Open another terminal with the same container

    ```console
    sudo docker exec -it 45ce850560c3 /bin/bash
    ```

### Using docker container

7) On this new docker container session use these commands to run the lane detector
    ```console
    cd /opt/nvidia/deepstream/deepstream/mqtt-example
    ```

    ```console
    python3 main.py
    ```