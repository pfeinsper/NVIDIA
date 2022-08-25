import cv2
import numpy as np
from keras.models import load_model
from PIL import Image 
from keras.preprocessing import image

focused_signs = {
    "proibido_virar_a_direita": 0,
    "proibido_virar_a_esquerda": 1,
    "pare": 2,
    "velocidade_maxima_40_km": 3,
    "velocidade_maxima_60_km": 4
}

image_x, image_y = 32,32


classifier = load_model("LeNet5_BRA.model")
    
cam = cv2.VideoCapture(0)

img_counter = 0

img_text = ['','']
while True:
    ret, frame = cam.read()
    img = cv2.rectangle(frame, (0,320),(0,320), (0,255,0), thickness=2, lineType=8, shift=0)

    imcrop = img[0:320, 0:320]
        
    cv2.putText(frame, str(img_text[1]), (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("Sinal", frame)
    cv2.imshow("LeNet5-BRA", imcrop)
            
    image = cv2.resize(imcrop, (image_x, image_y))
    image = np.expand_dims(image, axis = 0)
    result = classifier.predict(image)
    
    i = np.argmax(result)
    img_text = [result, list(focused_signs.keys())[i]]
        
    if cv2.waitKey(1) == 113:
        break

cam.release()
cv2.destroyAllWindows()