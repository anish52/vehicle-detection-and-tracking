# -*- coding: utf-8 -*-
import time
import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import Vehicle
import math

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = InteractiveSession(config=config)

font = cv2.FONT_HERSHEY_SIMPLEX 
from motrackers.detectors import TF_SSDMobileNetV2
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
#from motrackers.utils import draw_track
# ============================================= Loading the model ==============================================
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  #base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
#TEST_IMAGE_PATHS
INPUT_VIDEO_PATH = r'D:\Skellam\Data\input.mp4'
OUTPUT_VIDEO_DIR = r'D:\Skellam\Data\out\tracking'

model_name = 'faster_rcnn_resnet101_coco_2018_01_28'
detection_model = load_model(model_name)

print(detection_model.inputs)
detection_model.output_dtypes
# ============================================================================================================
# ============================================ Loading the tracker =========================================
#chosen_tracker = 'CentroidKF_Tracker'
chosen_tracker = 'CentroidTracker'
if chosen_tracker == 'CentroidTracker':
    tracker = CentroidTracker(max_lost=3, tracker_output_format='mot_challenge', exit_point=(320,420), exit_direction='Down')
elif chosen_tracker == 'CentroidKF_Tracker':
    tracker = CentroidKF_Tracker(max_lost=3, centroid_distance_threshold=300., tracker_output_format='mot_challenge')
elif chosen_tracker == 'SORT':
    tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
elif chosen_tracker == 'IOUTracker':
    tracker = IOUTracker(max_lost=3, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                         tracker_output_format='mot_challenge')
else:
    print("Please choose one tracker from the above list.")
# ============================================================================================================

# ============================================= Reading the video ============================================
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS),\
                                    cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frames_count = int(frames_count)
width = int(width)
height = int(height)
print("Reading video file...")
print("Number of frames: {}\nFPS: {:.2f}\nWidth of each frame: {}\nHeight of each frame: {}".\
      format(frames_count, fps, width, height))
# ============================================================================================================
# ============================= Define the codec and create VideoWriter object ===============================
stride = 30
fps_out = round(fps)//stride
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_DIR+'\output_indexed_'+chosen_tracker+str(fps_out)+'fps.avi', fourcc, fps_out, (width,height))
###out = cv2.VideoWriter(OUTPUT_VIDEO_DIR+'\Full_output_video.avi', fourcc, fps_out, (width,height))
# ============================================================================================================

def format2box(output_dict, width, height):
  '''
  Returns
  ----------
  bboxes : numpy.ndarray or list
      List of bounding boxes detected in the current frame. Each element of the list represent
      coordinates of bounding box as tuple `(top-left-x, top-left-y, width, height)`.
  detection_scores: numpy.ndarray or list
      List of detection scores (probability) of each detected object.
  class_ids : numpy.ndarray or list
      List of class_ids (int) corresponding to labels of the detected object. Default is `None`.
  '''
  bboxes, confidences, class_ids = [], [], [] 
  
  for i in range(output_dict['num_detections']):
    y1, x1, y2, x2 = output_dict['detection_boxes'][i]
    y1, x1, y2, x2 = int(y1*height), int(x1*width), int(y2*height), int(x2*width)
    bboxes.append([x1, y1, x2-x1, y2-y1])
    class_ids.append(output_dict['detection_classes'][i])
    confidences.append(output_dict['detection_scores'][i])

  bboxes = np.array(bboxes, dtype='int')
  class_ids = np.array(class_ids, dtype='int')
  confidences = np.array(confidences)
  return bboxes, confidences, class_ids

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  img = Image.open(image_path)
  #size = 128, 128
  #img.thumbnail(size, Image.ANTIALIAS)
  image_np = np.array(img)
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  out_img = Image.fromarray(image_np)
  display(out_img)
  out_img.show()


def process_tracks(tracks):
    tracks = np.array(tracks)
    boxes, classes, scores, ids = [], [], [], []
    for tracker_obj in tracks:
        id_, xmin, ymin, width, height, score = tracker_obj[1:7]
        boxes.append([ymin, xmin, ymin+height, xmin+width])
        classes.append(3)
        scores.append(score)
        ids.append(id_)

    return np.array(boxes), np.array(classes, dtype=int), np.array(scores), np.array(ids, dtype=int)


def run_inference_for_video(model):
    n_frames = 0
    while(True):
        # Capture frame-by-frame
        ret,image_np = cap.read()
        if ret == False:    break
        n_frames += 1
        if n_frames % 100 == 0:
            print(n_frames)
        if n_frames % stride != 0:
            continue
            #break
        # result image with boxes and labels on it.
        #img = Image.open(image_path)
        #size = 128, 128
        #img.thumbnail(size, Image.ANTIALIAS)
        image_np = np.array(image_np)
        image_np_with_detections = image_np.copy()

        # Selecting the ROI (Masking all other lanes except 'Drivethru' lane)
        external_poly = np.array( [[[0,0],[0,340],[640,80],[640,0]]], dtype=np.int32)
        cv2.fillPoly( image_np , external_poly, (0,0,0) )
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)

        output_dict2 = {'detection_boxes':[], 'detection_classes':[], 'detection_scores':[], 'num_detections':0}
        for i in range(output_dict['num_detections']):
          if (output_dict['detection_classes'][i] == 3 or output_dict['detection_classes'][i] == 8) and output_dict['detection_scores'][i] >= 0.64:
            output_dict2['detection_boxes'].append(output_dict['detection_boxes'][i])
            output_dict2['detection_classes'].append(output_dict['detection_classes'][i])
            output_dict2['detection_scores'].append(output_dict['detection_scores'][i])
            output_dict2['num_detections'] += 1
   
        cv2.rectangle(image_np_with_detections, (45, 45), (290, 65), (0,0,0), -1)
        cv2.putText(image_np_with_detections, "Number of cars detected : "+str(output_dict2['num_detections']),(50,60),font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # ======================================== Formatting for Tracking ============================================
        bboxes, confidences, class_ids = format2box(output_dict2, width, height)
        # ============================================= Tracking =====================================================
        tracks = tracker.update(bboxes, confidences, class_ids)
        
        # ============================================================================================================
        
        # Writing logs to file
        # print("="*25)
        # print(str(tracks))
        # print("="*25+"\n")
        with open(OUTPUT_VIDEO_DIR+"\logs_"+str(fps_out)+"fps.txt", "a") as f:
          f.write(str(n_frames)+": "+str(tracks)+"\n")
          

        boxes, classes, scores, ids =  process_tracks(tracks)
    
        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     image_np_with_detections,
        #     np.array(output_dict2['detection_boxes']),
        #     np.array(output_dict2['detection_classes']),
        #     np.array(output_dict2['detection_scores']),
        #     category_index,
        #     instance_masks=output_dict.get('detection_masks_reframed', None),
        #     use_normalized_coordinates=True,
        #     track_ids = ids,
        #     skip_scores=True,#removes scores
        #     skip_labels=True,#removes lables
        #     line_thickness=2)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes,
            classes,
            scores,
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=False,
            track_ids = ids,
            skip_scores=True,#removes scores
            skip_labels=True,#removes lables
            line_thickness=2)

        # Display the resulting frame
        out.write(image_np_with_detections)
        # cv2.imshow('frame',image_np_with_detections)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
start = time.time()
run_inference_for_video(detection_model)
span = time.time() - start
print("Model Name: {}".format(model_name))
print("Run Time: {}mins {}secs".format(int(span//60),int(span%60)))