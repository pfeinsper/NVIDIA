# Broker MQTT

Currently, in the software architecture scenario, it is very common that the architecture chosen for communication between different applications is the Publisher / Subscriber. A type of asynchronous messaging architecture that uses brokers and prioritizes the independence of a specific application that does not necessarily depend on the response of the other to function. Brokers are components of the messaging cluster, which allow communication between services with the guarantee of data integrity. So the implementation of a broker for asynchronous communication will be efficient to transfer the processing and computational resources allocated in the Jetson Nano to a local laptop with greater processing power.

!!! note  
    In this type of asynchronous architecture, the publisher is the service that produces the messages in the broker, indicating that a new event has happened. In the case of the project, JetBot is responsible for sending the image update event processed by the Detection, Classification and Autopilot service. The subscriber consumes messages from the broker and starts the process flow according to its mission.

So a very useful tool for the project was to use the [Node-Red](https://nodered.org/) with the [MQTT protocol](https://mqtt.org/) for IoT messaging communication between the robot and the classifier. 

The archtecture of the IoT messaging can be seen on the figure below:

![MQTT messaging](https://github.com/pfeinsper/NVIDIA/blob/gh-pages/images/nodered.png?raw=true)

The block in blue, named “Sensor 1“ is an ‘inject’ block, which sends a msg.payload and a msg.topic, the message content and its origin, respectively. The message sent, msg.payload, can be in different formats, including JSON, which has the greatest compatibility with Python, which is the language being used to develop the project.

The purple blocks refer to MQTT. For testing purposes, the server being used is 'localhost' on port 1883. The 'MQTT-OUT' blocks are the ones that receive the message coming from the system, while the 'MQTT-IN' gets the information from of the referred port, in this case, the 'localhost' itself.

Finally, the green block refers to the 'Debug' of the model, where the information that arrives in this block is presented in the terminal. Below, each of the nodes will be presented and, finally, the message information in the terminal when receiving the input from the 'inject'.

Above, the node of the 'inject' block is shown, in which the msg.payload and msg.topic are determined, as mentioned above. In this case, the message is in JSON format and its content is “Test”.

Then the MQTT-IN node, which subscribes to a message from a specific topic. While MQTT-OUT, publishes a message it received from the broker.

!!! warning "Remember"

    The images obtained by the JetBot, as it travels along the runway, will be sent periodically to the object detection system (signs, people, etc). In this way, detection occurs simultaneously with the 'road following' protocol, mentioned above. Thus, the system latency will be lower, allowing for better iteration and faster reaction time of the JetBot, something essential for an autonomous vehicle.

    The message sent by MQTT will be a JSON with the classification of the detected board, a 'STOP' board, for example. Thus, upon receiving this input, the JetBot's navigation system will be more agile when taking the action of turning off the engines and stopping the movement, according to the signal. If no card is detected, JetBot will continue with the 'Road Following' system normally.