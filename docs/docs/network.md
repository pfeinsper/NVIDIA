# Neural Networks

In this part of the project the neural networks system is basically a detector. This detector network is capable of identifing the traffic signs of the road, crop the traffic sign frame and classify it as a specific class, so that the network model classifies and detects the roadsign and send the information to the Broker MQTT that comunicates with the robot infinite state machine. 

## Model Overview

This neural network model is based on a object detection model from Nvidia GPU Cloud called [DashCamNet]("https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet"), which is built based on Nvidia [DetectNet_v2]("https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tao_detectnet") detector which uses ResNet18 and since it is a object detection model, the output is a bounding-box on the input image surrounding the object detected, the bouding-box is predicted by calculating the x center, y center, width and height from the object, not to mention the confidence value from the output class is returned as well.

The model classes identified are:

- pare (Indicating the stop sign)
- 40 (Indicating the 40km/h traffic sign)
- 60 (Indicating the 60km/h traffic sign)

## Building the model

The model was created using the transfer learning toolkit from Nvidia called [TAO Toolkit]("https://developer.nvidia.com/tao-toolkit") with a [Roboflow Dataset]("https://roboflow.com/"). The raise of the dataset is documented on the [Dataset]("") section. The main reference used for the traffic sign detector was based on an article called [Building Custom Computer Vision Models with NVIDIA TAO Toolkit and Roboflow]("https://blog.roboflow.com/nvidia-tao-toolkit-roboflow/") and a repository [tao-toolkit-with-roboflow]("https://github.com/roboflow-ai/tao-toolkit-with-roboflow") which used TAO toolkit to create a yolo_v4 based model using transfer learning.

In our case the steps were the same but the neural networks detector used was DetectNet_v2 instead of yolo_v4.

The notebook from our project is on Github on this link:

- **[Project Notebook]("https://github.com/pfeinsper/NVIDIA/blob/main/transfer-learning/transfer_learning_tutorial.ipynb")**

## Model Usage

If you want to just use our model, you can just download these files below on your Jetson device, on the other hand, if you want to build your own model you can follow the Detailed Transfer learning section below:

| Files            | Links                                |
| ----------- -----| ------------------------------------ |
| Etlt model       |  [https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J")|
| Labels file      |  [https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J")|
| Calibration file |  [https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J")|
| Engine file      |  [https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J")|

After downloading these files you need to create a `config_primary.txt` file with the content below:

```txt

[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
int8-calib-file=<Path to optional INT8 calibration cache>
labelfile-path=<Path to labels.txt>
tlt-encoded-model=<Path to ETLT model>
tlt-model-key=<Key to decrypt the model>
infer-dims=c;h;w # where c = number of channels, h = height of the model input, w = width of model input
force-implicit-batch-dim=1
batch-size=1
network-mode=1
num-detected-classes=3
interval=0
gie-unique-id=1
output-blob-names=conv2d_bbox;conv2d_cov/Sigmoid
#scaling-filter=0
#scaling-compute-hw=0

[class-attrs-all]
pre-cluster-threshold=0.2
eps=0.2
group-threshold=1
```

Create a python file called `deepstream_usb_camera_with_custom_model.py` to run the model:

!!! danger "REMEMBER"

    For this step you need to have the Broker MQTT, Deepstream and your current IP ADDRESS.

```python
#!/usr/bin/env python3
from paho.mqtt import client as mqtt_client
import sys
sys.path.append('../')
import gi
import time
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import random

import pyds

PGIE_CLASS_ID_PARE = 0
PGIE_CLASS_ID_40 = 1
PGIE_CLASS_ID_60 = 2


broker = 'YOUR_MACHINE_IP_ADDRESS'
port = 1884
topic = "python/mqtt"
client_id = f'python-mqtt-{random.randint(0, 1000)}'
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client, msg):
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send {msg}")
    else:
        print(f"Failed to send message to topic {topic}")

client = connect_mqtt()
client.loop_start()

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_PARE:0,
        PGIE_CLASS_ID_60:0,
        PGIE_CLASS_ID_40:0
    }
    num_rects=0
    x_left = 0
    x_right = 0
    y_top = 0
    y_bottom = 0 
    area = 0
    roadsign_detected = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                print('Detected "' + obj_meta.obj_label + '" with ID: ' + str(obj_meta.object_id))

            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        
        if obj_counter[PGIE_CLASS_ID_ROADSIGN] > 0:
            x_left = int(obj_meta.rect_params.left)
            x_right = int(obj_meta.rect_params.left + obj_meta.rect_params.width)
            y_top = int(obj_meta.rect_params.top)
            y_bottom =  int(obj_meta.rect_params.top + obj_meta.rect_params.height)
            area = abs(x_right - x_left)* abs(y_bottom - y_top)
            if area > 38000:
                roadsign_detected = 1
        else:
            roadsign_detected = 0
            
        py_nvosd_text_params.display_text = "DeepStream {0}".format(roadsign_detected)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        publish(client, pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <v4l2-device-path>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")


    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)


    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Use nvtracker to give objects unique-ids
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not pgie:
        sys.stderr.write(" Unable to create tracker \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Playing cam %s " %args[1])
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    source.set_property('device', args[1])
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "config_primary.txt")
    #Set properties of tracker from tracker_config
    config = configparser.ConfigParser()
    config.read('../deepstream-test2/dstest2_tracker_config.txt')
    config.sections()
    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
    # Set sync = false to avoid late frame drops at the display-sink
    sink.set_property('sync', False)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)

    # we link the elements together
    # v4l2src -> nvvideoconvert -> mux -> 
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
```

Finally you can run the python specifing the config file and the usb camera path:

**Normally you can test it using `/dev/video0` for the usb camera path**

```bash
python3 deepstream_usb_camera_with_custom_model.py /dev/{USB_CAMERA_VIDEO_PATH}
```

## Detailed Transfer Learning

Since you want to build a custom model by yourself you can follow the notebooks from the links below and download the files referenced, so that you can follow the correct steps:

- [tao-toolkit-with-roboflow]("https://github.com/roboflow-ai/tao-toolkit-with-roboflow") 
- [Jetbot transfer learning]("https://github.com/pfeinsper/NVIDIA/blob/main/transfer-learning/transfer_learning_tutorial.ipynb")


| Files                         | Links                                                                                                                                                     |
| ------------------------------| --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| experiment_spec.json          |  ["https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J"]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J") |
| detectnet_v2_inference.txt    |  ["https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J"]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J") |
| detectnet_v2_inference_etlt   |  ["https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J"]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J") |
| detectnet_v2_retrain_resnet18 |  ["https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J"]("https://drive.google.com/drive/folders/1TCEeig-Y4BD1gVVE5CkSWjv4DcaY7r4J") |
<!-- ```bash
pip install nvidia-tao
pip install roboflow
```

After doing that you need to export the roboflow dataset using the COCO format, you can do it manually or using python code as shown below:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("ROBOFLOW_WORKSPACE").project("ROBOFLOW_PROJECT_NAME")
dataset = project.version(1).download("coco")
```

Since TAO toolkit uses a docker container to run the traning, testing, inference and other machine learning parts, you need to specify the directory path from your project and dataset so that the docker container can access all your project files, for this you can use this python code below:

```python
import os
import json

%env KEY=tlt_encode
%env NUM_GPUS=1

os.environ["LOCAL_PROJECT_DIR"] = "PATH_FROM_YOUR_PROJECT"

os.environ["LOCAL_DATA_DIR"] = os.path.join(
    os.getenv("LOCAL_PROJECT_DIR", os.getcwd()),
    "YOUR_DATASET_DIRECTORY"
)
z
mounts_file = os.path.expanduser("~/.tao_mounts.json")

# Define the dictionary with the mapped drives
drive_map = {
    "Mounts": [
        # Mapping the data directory
        {
            "source": os.environ["LOCAL_PROJECT_DIR"],
            "destination": "/workspace/project"
        },
    ],
    "DockerOptions": {
        # preserving the same permissions with the docker as in host machine.
        "user": "{}:{}".format(os.getuid(), os.getgid())
    }
}

# Writing the mounts file.
with open(mounts_file, "w") as mfile:
    json.dump(drive_map, mfile, indent=4)

```

```python
dataset_config = """
coco_config {
  root_directory_path: "/workspace/project/YOUR_DATASET_DIRECTORY"
  img_dir_names: ["test", "train", "valid"]
  annotation_files: [ "test/_annotations.coco.json", "train/_annotations.coco.json", "valid/_annotations.coco.json"]
  num_partitions: 3
  num_shards: [1, 72, 2]
}
image_directory_path: "/workspace/project/YOUR_DATASET_DIRECTORY"
"""

# write the dataset config to a JSON file
file_object = open('coco_config.json', 'w')
file_object.write(dataset_config)
file_object.close()

#Verify the coco_config.json file
!cat coco_config.json
```

You can verify that the docker container can access the project files using this code below:
```python
!tao bpnet run ls /workspace/project/YOUR_DATASET_DIRECTORY/test
```

The next part you need to  -->
