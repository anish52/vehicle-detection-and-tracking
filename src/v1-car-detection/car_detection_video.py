# -*- coding: utf-8 -*-

import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

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
INPUT_VIDEO_PATH = 'D:\Skellam\Data\input.mp4'
OUTPUT_VIDEO_DIR = 'D:\Skellam\Data\out'

#model_name = 'faster_rcnn_resnet101_coco_2018_01_28'
model_name = 'faster_rcnn_resnet101_kitti_2018_01_28'
#model_name = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
#model_name = 'faster_rcnn_nas_coco_24_10_2017'(best)
#model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
detection_model = load_model(model_name)

print(detection_model.inputs)
detection_model.output_dtypes

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
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_DIR+'\output_'+model_name+'.avi', fourcc, round(fps), (width,height))
# ============================================================================================================


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
  
def run_inference_for_video(model):
    n_frames = 0
    while(True):
        # Capture frame-by-frame
        ret,image_np = cap.read()
        if ret == False:    break
        n_frames += 1
        if n_frames % 100 == 0:
            print(n_frames)
            #break
        # result image with boxes and labels on it.
        #img = Image.open(image_path)
        #size = 128, 128
        #img.thumbnail(size, Image.ANTIALIAS)
        image_np = np.array(image_np)
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

        # Display the resulting frame
        out.write(image_np)
        #cv2.imshow('frame',image_np_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
run_inference_for_video(detection_model)