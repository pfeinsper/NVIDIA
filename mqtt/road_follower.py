import cv2
import numpy as np
import math
import auxiliar as aux
import random
import time
from paho.mqtt import client as mqtt_client

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


def publish(client, msg_string):
    msg = f"{msg_string}"
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")

client = connect_mqtt()
client.loop_start()

video1 = cv2.VideoCapture(4)
coef_angular = 0
coef_angular_anterior = 0
x_1 = 0
x_2 = 0
y_1 = 0
y_2 = 0
x_r1 = 0
x_r2 = 0
y_r1 = 0
y_r2 = 0
x_l1 = 0
x_l2 = 0
y_l1 = 0
y_l2 = 0
h_l = 1.0
h_r = 2.0
m_l = 2.0
m_r = 1.0
rx_mean = []
ry_mean = []
rx = 0
ry = 0
kaka = 0
while(True):

    ret, frame = video1.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = "#eb4034"
    white_1, white_2 = aux.ranges(white)
    branco_1 = np.array([20, 0, 20], dtype=np.uint8)
    branco_2 = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, branco_1, branco_2)
    edges = cv2.Canny(gray,50,150,apertureSize = 3) 
    lines = cv2.HoughLines(mask_white,1,np.pi/180, 200) 
    
    if type(lines) != type(None):
        
        for i in range(0, len(lines)): 
            r = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta) 

            b = np.sin(theta) 

            x0 = a*r 

            y0 = b*r 

            x1 = int(x0 + 1000*(-b)) 

            y1 = int(y0 + 1000*(a)) 
        
            x2 = int(x0 - 1000*(-b)) 

            y2 = int(y0 - 1000*(a)) 
            
            delta_x = x1-x2
            delta_y = y1-y2
            if delta_x == 0:
                coef_angular = 0
            else:
                coef_angular = delta_y/delta_x
            modulo_coef = ((coef_angular - coef_angular_anterior)**2)**0.5
            if modulo_coef > 0.3:
                coef_angular_anterior = coef_angular
            coef_angular = float(coef_angular)
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
            
            if coef_angular > 0.62 and coef_angular < 1.3:
                x_r1 = x1
                x_r2 = x2
                y_r1 = y1
                y_r2 = y2
                m_r = coef_angular
                h_r = y_r1 - m_r*x_r1
            elif coef_angular > -1.3 and coef_angular < -0.62:
                x_l1 = x1
                x_l2 = x2
                y_l1 = y1
                y_l2 = y2
                m_l = coef_angular
                h_l = y_l1 - m_l*x_l1

        cv2.line(mask_white,(x_l1, y_l1), (x_l2,y_l2), (50,0,255),2) 
        cv2.line(mask_white,(x_r1,y_r1), (x_r2,y_r2), (50,0,255),2)
        xi = (h_r-h_l)/(m_l-m_r)
        yi = (m_l*xi + h_l)
        xii = int(xi)
        yii = int(yi)
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
        if kaka < 20:
            rx_mean.append(xii)
            ry_mean.append(yii)
            rx = np.mean(rx_mean)
            ry = np.mean(ry_mean)
            rx = int(rx)
            ry = int(ry)
            kaka += 1
        if kaka >= 20:
            kaka = 0
            rx_mean = []
            ry_mean = []
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
        cv2.circle(mask_white, (xii, yii), 2, (255,255,255), 2)
        cv2.circle(mask_white, (xii, yii), 10, (255,255,255), 2)
        cv2.circle(mask_white, (rx, ry), 50, (255,255,255), 2)
        publish(client, f"x: {rx} \ny: {ry}")
    cv2.imshow('mask_white', mask_white)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video1.release()
cv2.destroyAllWindows()
