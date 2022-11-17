# Docker Setup

So in this page you will need to setup the docker container by yourself and install all the dependencies for the project on your Jetson device.

For this you can use the docker container from Nvidia GPU Cloud, called [DeepStream-l4t](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream-l4t). So to start, you will need to run the docker image from NGC as shown below.

```bash
xhost +

docker run -it --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.1 -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/deepstream-l4t:6.1.1-base
```

Clone the version v1.1.1 of [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps) Github repository that works specifically with the Jetpack 4.6.1: 

```bash
cd /opt/nvidia/deepstream/deepstream/sources/apps

git clone -b v1.1.1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

cd deepstream_python_apps/
```

Install some dependencies and execute some setup files
```
sudo apt install -y git python-dev python3 python3-pip python3.6-dev python3.8-dev cmake g++ build-essential     libglib2.0-dev libglib2.0-dev-bin python-gi-dev libtool m4 autoconf automake

git submodule update --init

sudo apt-get install --reinstall ca-certificates

sudo apt-get install python3.6-dev libpython3.6-dev

cd 3rdparty/gst-python/

./autogen.sh PYTHON=python3.6

sudo make install
```

Create some directories and install more features.

```bash
cd /opt/nvidia/deepstream/deepstream/sources/apps/deepstream_python_apps/bindings/

mkdir build

cd build

cmake ..  -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=6 -DPIP_PLATFORM=linux_aarch64 -DDS_PATH=/opt/nvidia/deepstream/deepstream-6.0/

make

sudo apt install libgirepository1.0-dev libcairo2-dev

pip3 install ./pyds-1.1.1-py3-none-linux_aarch64.whl 
```

Then you should change the directory files from the deepstream config file:

```bash
sudo apt install nano

sudo nano /opt/nvidia/deepstream/deepstream/sources/apps/deepstream_python_apps/apps/deepstream-test1/dstest1_pgie_config.txt
```

In this file you should substitute these variables shown below on the `dstest1_pgie_config.txt` file.


```txt
model-file=../../../../samples/models/Primary_Detector/resnet10.caffemodel
proto-file=../../../../samples/models/Primary_Detector/resnet10.prototxt
model-engine-file=../../../../samples/models/Primary_Detector/resnet10.caffemodel_b1_gpu0_int8.engine
labelfile-path=../../../../samples/models/Primary_Detector/labels.txt
int8-calib-file=../../../../samples/models/Primary_Detector/cal_trt.bin
```

Now the setup is done and you should test if DeepStream is working correctly just run this commands and this may take a few minutes to execute, but a new window should appear in your Jetson device.

```bash
cd /opt/nvidia/deepstream/deepstream/sources/apps/deepstream_python_apps/apps/deepstream-test1

python3 deepstream_test_1.py /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264 
```