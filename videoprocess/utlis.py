import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import cv2
import numpy as np
import mediapipe as mp
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
from PIL import Image as im



MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')

model_path = os.path.abspath(model_path) # './detector.tflite')




def detect_face(img_obj):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions   
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a face detector instance with the image mode:
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.3)
    

    with FaceDetector.create_from_options(options) as detector:
        detection_result = face_detector_result = detector.detect(img_obj)

    return detection_result

def mask_face(img_obj, detection_result):
    image_copy = np.copy(img_obj.numpy_view())
    annotated_image = image_copy.copy()
    height, width, _ = image_copy.shape
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        # point_0 = bbox.origin_x, bbox.origin_y
        # point_1 = bbox.origin_x, bbox.origin_y + bbox.height
        # point_2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        # point_3 = bbox.origin_x + bbox.width, bbox.origin_y 
        mask = np.zeros((height, width), np.uint8)
        convexhull = np.zeros((4, 1, 2), np.int32)
        convexhull[0][0][0] = bbox.origin_x
        convexhull[0][0][1] = bbox.origin_y

        convexhull[1][0][0] = bbox.origin_x
        convexhull[1][0][1] = bbox.origin_y + bbox.height

        convexhull[2][0][0] = bbox.origin_x + bbox.width
        convexhull[2][0][1] = bbox.origin_y + bbox.height
    
        convexhull[3][0][0] = bbox.origin_x + bbox.width
        convexhull[3][0][1] = bbox.origin_y 

        cv2.fillConvexPoly(mask, convexhull, 255)

    frame_copy = cv2.blur(annotated_image, (27, 27))
    face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(annotated_image, annotated_image, mask=background_mask)
    result = cv2.add(background, face_extracted)

    return result

def store_png(img_mat, name, path=None):
    data = im.fromarray(img_mat)
    if path != None:
        filename = os.path.join(path, f'{name}.png')
    else:
        filename = f'{name}.png'
    data.save(filename)