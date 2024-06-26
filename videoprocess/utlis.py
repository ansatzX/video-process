import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image as im
import cv2
import numpy as np
import mediapipe as mp
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import location_data_pb2

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
BG_COLOR = (192, 192, 192) 
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

blaze_face_model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')

blaze_face_model_path = os.path.abspath(blaze_face_model_path) # './detector.tflite')




def detect_face_by_blaze_face(img_obj):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions   
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a face detector instance with the image mode:
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=blaze_face_model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.3)
    

    with FaceDetector.create_from_options(options) as detector:
        detection_result = face_detector_result = detector.detect(img_obj)

    if len(detection_result.detections) == 0:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=blaze_face_model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.15)
        with FaceDetector.create_from_options(options) as detector:
            detection_result = face_detector_result = detector.detect(img_obj)

    return detection_result

def detect_face_by_face_solution(img_file):
    """
    need web-download
    """

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image = cv2.imread(img_file)
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return results

def detect_face_by_holistic(img_file):
    """
    need web-download
    """

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:
        image = cv2.imread(img_file)
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return results

def mask_face(img_file, detection_result, model_name):
    img_obj = mp.Image.create_from_file(img_file)
    cv_image_obj = cv2.imread(img_file)
    cv_image_mat = cv_image_obj.copy()
    

    image_copy = np.copy(img_obj.numpy_view())
    annotated_image = image_copy.copy()
    height, width, _ = image_copy.shape
    mask = np.zeros((height, width), np.uint8)
    if model_name == 'blaze' :
        for detection in detection_result.detections:
            bbox = detection.bounding_box

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

    elif model_name == 'face_solution' :
        if detection_result.detections is not None:
            
            for detection in detection_result.detections:

                convexhull = np.zeros((5, 1, 2), np.int32)
                convexhull[0][0][0] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION).x * width)
                convexhull[0][0][1] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION).y * height)

                convexhull[1][0][0] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).x * width)
                convexhull[1][0][1] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).y * height)

                convexhull[2][0][0] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION).x * width)
                convexhull[2][0][1] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION).y * height)

                convexhull[3][0][0] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE).x * width)
                convexhull[3][0][1] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE).y * height)

                convexhull[4][0][0] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE).x * width)
                convexhull[4][0][1] = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE).y * height)

                cv2.fillConvexPoly(mask, convexhull, 255)

                bbox_drawing_spec = mp_drawing.DrawingSpec()
                if not detection.location_data:
                    continue
                if cv_image_mat.shape[2] != _BGR_CHANNELS:
                    raise ValueError('Input image must contain three channel bgr data.')
                image_rows, image_cols, _ = cv_image_obj.shape
                # image_rows, image_cols, _ = image.sh       
                location = detection.location_data                                         
                if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                  raise ValueError(
                      'LocationData must be relative for this drawing funtion to work.')
  
                if not location.HasField('relative_bounding_box'):
                  continue
                relative_bounding_box = location.relative_bounding_box

                convexhull = np.zeros((4, 1, 2), np.int32)
                convexhull[0][0][0] = int(relative_bounding_box.xmin * width )
                convexhull[0][0][1] = int(relative_bounding_box.ymin * height )

                convexhull[1][0][0] = int(relative_bounding_box.xmin * width )
                convexhull[1][0][1] = int((relative_bounding_box.ymin  + relative_bounding_box.height)* height)

                convexhull[2][0][0] = int((relative_bounding_box.xmin + relative_bounding_box.width)* width)
                convexhull[2][0][1] = int((relative_bounding_box.ymin + relative_bounding_box.height)* height)
    
                convexhull[3][0][0] = int((relative_bounding_box.xmin + relative_bounding_box.width)* width)
                convexhull[3][0][1] = int(relative_bounding_box.ymin * height)
                # print(convexhull)
                rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                        image_rows)
                rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                        image_rows)
                cv2.rectangle(mask, rect_start_point, rect_end_point,
                    bbox_drawing_spec.color, bbox_drawing_spec.thickness)
                
                cv2.fillConvexPoly(mask, convexhull, 255)




    elif model_name == 'holistic' :
        if detection_result.pose_landmarks.landmark is not None:
            convexhull = np.zeros((6, 1, 2), np.int32)
            convexhull[0][0][0] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * width)
            convexhull[0][0][1] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * height)

            convexhull[1][0][0] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x * width)
            convexhull[1][0][1] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y * height)

            convexhull[2][0][0] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x * width)
            convexhull[2][0][1] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y * height)

            convexhull[3][0][0] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x * width)
            convexhull[3][0][1] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y * height)

            convexhull[4][0][0] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * width)
            convexhull[4][0][1] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * height)

            convexhull[5][0][0] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * width)
            convexhull[5][0][1] = int(detection_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * height)
            cv2.fillConvexPoly(mask, convexhull, 255)

    frame_copy = cv2.blur(annotated_image, (27, 27))
    face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(annotated_image, annotated_image, mask=background_mask)
    masked_mat = cv2.add(background, face_extracted)

    return masked_mat

def store_png(img_mat, name, path=None):
    data = im.fromarray(img_mat)
    if path != None:
        filename = os.path.join(path, f'{name}.png')
    else:
        filename = f'{name}.png'
    data.save(filename)