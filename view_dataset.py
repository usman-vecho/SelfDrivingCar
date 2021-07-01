import cv2
import numpy as np
import tensorflow as tf

train_data = list(np.load('Dataset.npy' , allow_pickle = True))
c = 0
for image,out in train_data:
    image = cv2.resize(image , (150,150))
    cv2.imshow('img',cv2.resize(image,(500,500)))
    print(out)
    cv2.waitKey(0)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        break
    
cv2.destroyAllWindows()
print(c)



