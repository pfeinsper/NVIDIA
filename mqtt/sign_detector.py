# import the necessary packages
import random
import time
from paho.mqtt import client as mqtt_client
import cv2
import numpy as np
import time

broker = '10.102.20.105'
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


def publish(client):
    msg = "STOP"
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send {msg}")
    else:
        print(f"Failed to send message to topic {topic}")

client = connect_mqtt()
client.loop_start()

# load the COCO class names
with open('./src/coco.names', 'r') as f:
    class_names = f.read().split('\n')
    
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
model = cv2.dnn.readNet(model='./src/yolov3-tiny.weights',
                        config='./src/yolov3-tiny.cfg')

layers = model.getLayerNames()
output_layers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(3)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    frame_id += 1
    height, width, channels = frame.shape
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # set the blob to the model
    model.setInput(blob)
    # forward pass through the model to carry out the detection
    output = model.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2 and (str(class_names[class_id]) == "stop sign"):
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                publish(client)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]
            color = COLORS[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 1)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 2)
    cv2.imshow('image', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
vs.release()
cv2.destroyAllWindows()
