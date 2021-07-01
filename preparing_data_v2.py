import urllib.request
import numpy as np
import cv2
import socket
import keyboard
import random
from hand_coded import HandCodedLaneFollower

num = random.randrange(1,100000)

url='http://192.168.1.102:8080/shot.jpg'

host = '192.168.1.106'
port = 9000
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((host,port))


lane_follower = HandCodedLaneFollower()
output = [0,1,0]
co = 0
data = []
while True:
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    img = cv2.resize(img  , (500,500))
    #img = img[::-1]
    #img = cv2.flip(img,1)
    image_ = cv2.resize(img , (224,224))
    cv2.imshow('img',cv2.resize(img,(400,400)))

    if keyboard.is_pressed('s'):
        s.send(bytes('S','utf-8'))
        output = [1,0,0]
    if keyboard.is_pressed('f'):
        s.send(bytes('F','utf-8'))
        output = [0,0,1]
    if keyboard.is_pressed('e'):
        s.send(bytes('E','utf-8'))
        output = [0,1,0]
    if keyboard.is_pressed('k'):
        s.send(bytes('K','utf-8'))
    if keyboard.is_pressed('l'):
        s.send(bytes('L','utf-8'))
        #cv2.destroyAllWindows()
    combo_image= lane_follower.follow_lane(img)
    print('Turning car at angle = ',lane_follower.curr_steering_angle)
    cv2.imshow("Road with Lane line", cv2.resize(combo_image,(700,700)))


    #data.append([image_,output])
    #if co == 200:
        #np.save('Car-dataset/{}.npy'.format(num),data)
    #    s.send(bytes('L','utf-8'))
    #    s.send(bytes('E','utf-8'))
    #    cv2.destroyAllWindows()
    #    import sys



    co+=1

    key = cv2.waitKey(33)
    if key==27:
        break
        cv2.destroyAllWindows()
#cv2.destroyAllWindows()





