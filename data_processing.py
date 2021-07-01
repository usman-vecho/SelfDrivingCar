import cv2
import numpy as np
#import tensorflow as tf

train_data = list(np.load('Final.npy' , allow_pickle = True))
print(len(train_data))
data = []

for image,out in train_data:
    cv2.imshow('img',image)
    print(out)
    cv2.waitKey(0)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        break
    
cv2.destroyAllWindows()



