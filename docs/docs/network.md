# Neural Networks

In this part of the project the neural networks are divided in two parts, the detector network and the classifier network. The detector network is capable of identifing the traffic signs of the road, crop the traffic sign frame and send the cropped image to the classifier, so that the classifier model classifies the roadsign and send it to the Broker MQTT. 

All these steps are shown in the figure below:

![Neural Networks Architecture](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/networks.png?raw=true)

## Detector

The detector network detects one or more physical objects from four categories within an image and returns a box around each object, as well as a category label for each object. 

The four categories of objects detected by this model are: 

- Cars
- Persons, 
- Road signs 
- Two-wheelers (Bycicles and Motorcycles)

This model is called [TrafficCamNet]("https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet"), this model can be found on the [Nvidia GPU Cloud Catalog]("https://catalog.ngc.nvidia.com/") and it is based on the Nvidia DetectNet_v2 and the ResNet18. The TrafficCamNet returns a balding-box with the confidence detections of the object, so that the user can see what the model detects. 

All these features can be seen in the video below:

<video width="480" height="240" controls>
  <source src="https://raw.githubusercontent.com/pfeinsper/NVIDIA/gh-pages/videos/trafficcamnet.webm"/>
</video>

## Classifier

![LeNet5 Topology](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/lenet.png?raw=true){ align=right }

The classifier model is based on [Eddie Forson]("https://towardsdatascience.com/recognizing-traffic-signs-with-over-98-accuracy-using-deep-learning-86737aedc2ab") article, that provides a 98% accuracy for traffic sign classifications with germany traffic signs. As the project needs to classify brazzilian traffic signs, the best option was to use a Transfer Learning with the Eddie Forson model that uses the [LeNet5]("https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342") as a CNN (Convolutional Neural Network) to be able to train the model with brazilian traffic signs insteat of using the germany ones. 

In the video below you can see the result of the LeNet5 classifier:

<video width="360" height="360" controls>
  <source src="https://raw.githubusercontent.com/pfeinsper/NVIDIA/gh-pages/videos/lenettest.mp4"/>
</video>