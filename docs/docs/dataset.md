# Building the Dataset

For the project dataset we uses [Roboflow Dataset](https://roboflow.com/), a online platform to build your own image dataset to work with computer vision models. In the case of the project we used roboflow to upload the traffic signs images and draw the bounding-box around the traffic signs for them to be identified on the training and testing from the model. 

To build the dataset we used the robot camera to take photos of the traffic signs on the test sight and used roboflow to draw the bounding box on the images to identify where the traffic sign is, you can see some examples of the images we've taken below.

___

### Traffic Sign Photos

<figure markdown>
  ![pare](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/pare.png?raw=true){ width="300" }
  <figcaption>Stop sign</figcaption>
</figure>
<figure markdown>
  ![40](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/40.png?raw=true){ width="300" }
  <figcaption>40km/h sign</figcaption>
</figure>
<figure markdown>
  ![60](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/60.png?raw=true){ width="300" }
  <figcaption>60km/h sign</figcaption>
</figure>
<figure markdown>
  ![right](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/right.png?raw=true){ width="300" }
  <figcaption>Turn right sign</figcaption>
</figure>
<figure markdown>
  ![left](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/left.png?raw=true){ width="300" }
  <figcaption>Turn left sign</figcaption>
</figure>

After uploading all traffic sign images we used roboflow to build a data augumentation growing the dataset with more images, in order to make the model more precise and robust, you can see the data augumentation page after uploading all the images, as shown below.

![https://blog.roboflow.com/content/images/size/w1000/2022/08/Screen-Shot-2022-08-11-at-1.20.08-AM.webp](https://blog.roboflow.com/content/images/size/w1000/2022/08/Screen-Shot-2022-08-11-at-1.20.08-AM.webp)

You can find all these steps on the "Building Custom Computer Vision Models with NVIDIA TAO Toolkit and Roboflow" tutorial, on the link below.

[https://blog.roboflow.com/nvidia-tao-toolkit-roboflow/](https://blog.roboflow.com/nvidia-tao-toolkit-roboflow/)
