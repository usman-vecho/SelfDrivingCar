import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 #Use OpenCV instead of Matplot
import socket
from collections import defaultdict
from io import StringIO
#from hand_coded import HandCodedLaneFollower
import keyboard
import urllib.request


#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'#'faster_rcnn_resnet101_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

NUM_CLASSES = 11


global detection_graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def run_inference_for_single_image(image, graph):
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict

from tensorflow import keras
import tensorflow as tf
import numpy as np
#import log

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)


INPUT_SHAPE = (-1, 150, 150, 3)



MODEL_NAME = 'T_m.model'
model_ = tf.keras.models.load_model(MODEL_NAME)


def predict_(image_arr):
    try:
        with session.as_default():
            with session.graph.as_default():
                #image_arr = np.array(image_arr).reshape(INPUT_SHAPE)
                predicted_labels = model_.predict(image_arr.reshape(INPUT_SHAPE))
                #print(predicted_labels)
                return predicted_labels
    except Exception as ex:
        print(str(ex))
        
url='http://192.168.1.103:8080/shot.jpg'
directions = ['left','forward','right']

host = '192.168.1.105'
port = 9000
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((host,port))


directions = ['left','forward','right']
s.send(bytes('K','utf-8'))
with detection_graph.as_default():
  with tf.Session() as sess:
    while True:


      imgResp=urllib.request.urlopen(url)
      imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
      image_np=cv2.imdecode(imgNp,-1)

      image = cv2.resize(image_np, (300, 300))
      img = image.copy()
      img = cv2.resize(img,(150,150))
      #cv2.imshow('img-',img)
      #combo_image= lane_follower.follow_lane(cv2.resize(img,(400,400)))
      #cv2.imshow('img',cv2.resize(combo_image,(400,400)))
      
      actual = predict_(img)
      pred = np.argmax(actual)
      

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

      output_dict = run_inference_for_single_image(image_np, detection_graph)
      vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)

        
      #cv2.putText(image_np,str(cats_[actual]),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))
      cv2.imshow('object detection', cv2.resize(image_np, (800,800)))
      key = cv2.waitKey(33)
      if key==27:
        cv2.destroyWindow()
        break
cap.release()

