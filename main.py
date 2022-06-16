from asyncio.windows_events import NULL
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import time
from vidgear.gears import CamGear

options = {"STREAM_RESOLUTION": "480p"}

LABEL_MAP = os.path.join('CAR_DETECTION_GRAPH', 'label_map.pbtxt')
CONFIG = os.path.join('CAR_DETECTION_GRAPH', 'pipeline.config')
CHECKPOINT = os.path.join('CAR_DETECTION_GRAPH', 'checkpoint',  'ckpt-0')

stream = CamGear(source='https://www.youtube.com/watch?v=1EiC9bvVGnk', stream_mode = True, logging=True, **options).start()

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP)
configs = config_util.get_configs_from_pipeline_file(CONFIG)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT)).expect_partial()

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def detect_models(image):
    image_np = np.array(image)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=15,
                min_score_thresh=.75,
                agnostic_mode=False)
    return image_np_with_detections

skip_frame =0

while True: 
    start_time = time.time()
    frame = stream.read()

    skip_frame+=1

    if frame is None:
        break

    img = frame
    height = img.shape[0]
    width = img.shape[1]
    img_cropped = img[20:height,0:width]

    if skip_frame<5:
        continue

    skip_frame=0

    cv2.imshow('car detection',  detect_models(img))
    cv2.imshow('test', img)

    print("FPS: ", 1.0 / (time.time() - start_time))
    cv2.waitKey(1)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break