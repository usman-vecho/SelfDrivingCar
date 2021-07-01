import urllib.request
import numpy as np
import cv2
import tensorflow as tf
import socket
import keyboard

url='http://192.168.1.103:8080/shot.jpg'
directions = ['left','forward','right']

host = '192.168.1.105'
port = 9000
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((host,port))

MODEL_NAME = 'T_m.model'
model = tf.keras.models.load_model(MODEL_NAME)


while True:
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    image_ = cv2.resize(img , (150,150))
    cv2.imshow('img',cv2.resize(img,(400,400)))
    #combo_image= lane_follower.follow_lane(cv2.resize(img,(400,400)))
    #cv2.imshow('img',cv2.resize(combo_image,(400,400)))
    pred = model.predict([image_.reshape(-1,150,150,3)])
    pred = np.argmax(pred)
    print(directions[pred])
    if pred == 0:
        s.send(bytes('S','utf-8'))
    if pred == 2:
        s.send(bytes('F','utf-8'))
    if pred == 1:
        s.send(bytes('E','utf-8'))
    if keyboard.is_pressed('k'):
        s.send(bytes('K','utf-8'))
    if keyboard.is_pressed('l'):
        s.send(bytes('L','utf-8'))
    if keyboard.is_pressed('b'):
        s.send(bytes('B','utf-8'))

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()