import cv2
import numpy as np
from math import tan, radians
from sklearn.linear_model import LinearRegression
from paho.mqtt import client as mqtt_client
import random


broker = '10.102.30.249'
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


class LaneDetector:
    def __init__(self):
        # self.video_path = video_path
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.mask = None
        self.contours = None
        self.m = None
        self.p1 = None
        self.p2 = None
        self.point_center = None 
    def start_detector(self, draw=True):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                print('no frame')
                break

            self.frame = self.frame[self.frame.shape[1]//6:, :]

            self.mask_yellow()
            self.get_contours()
            self.get_all_contours_centers()
            if len(self.centers) > 0:
                self.linear_regression()
                self.steer()

                if draw:
                    for center in self.centers:
                        cv2.circle(self.frame, center, 5, (0, 255, 255), -1)
                    cv2.circle(self.frame, self.centers[0], 5, (255, 0, 0), -1)
                    cv2.line(self.frame, self.p1, self.p2, (255, 255, 0), 3)
                    cv2.line(self.frame, (self.frame.shape[1]//2, 0), (self.frame.shape[1]//2, self.frame.shape[0]), (0, 255, 39), 3)
                    
            #cv2.imshow('frame', self.frame)
            #cv2.imshow("mask", self.mask)

            if cv2.waitKey(30) & 0xFF == ord('q'): break
        self.cap.release()
        cv2.destroyAllWindows()

    def mask_yellow(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([22, 50, 50])
        upper_yellow = np.array([36, 255, 255])
        self.mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel1 = np.ones((22,22),np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel1)
        #self.mask = cv2.blur(self.mask,(9,9))
        # kernel2 = np.ones((15,15),np.uint8)
        # self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel2)

    def get_contours(self):
        contourss, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.contours = [contour for contour in contourss if cv2.contourArea(contour) > 40]

    def get_all_contours_centers(self):
        self.centers = []
        for contour in self.contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                self.centers.append((cx, cy))

    def linear_regression(self):
        x = [point[0] for point in self.centers]
        y = [point[1] for point in self.centers]
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        self.m, h = model.coef_[0][0], model.intercept_[0]
        xMin = int(min(x))
        xMax = int(max(x))

        yMin = int(self.m*xMin + h)
        yMax = int(self.m*xMax + h)

        self.p1 = (xMin, yMin)
        self.p2 = (xMax, yMax)

    def steer(self):
        lim = 30
        center_index = len(self.centers)//2
        print(f'coord central = {self.centers[center_index]}')
        #cv2.circle(self.frame, self.centers[center_index], 10, (0, 255, 0), -1)
        self.k = 0
        if self.frame.shape[1]/2 - lim < self.centers[0][0] < self.frame.shape[1]/2 + lim:
            cv2.putText(self.frame, f'KEEP CENTER', (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            delta = self.centers[0][0] - self.frame.shape[1]/2
            max_delta = self.frame.shape[1]/2
            self.k = delta/max_delta * 0.2 + tan(radians(self.m))/360
            if self.k < 0:
                cv2.putText(self.frame, f'TURN LEFT {self.k:.3f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif self.k > 0:
                cv2.putText(self.frame, f'TURN RIGHT {self.k:.3f}', (340, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        publish(client, str(self.centers[center_index][0])+ ',' + str(self.centers[center_index][1]) + ',' + str(self.k))

if __name__ == '__main__':
    ld = LaneDetector()
    ld.start_detector(draw=False)
