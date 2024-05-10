from videoprocess.utlis import detect_face_by_blaze_face, detect_face_by_face_solution, detect_face_by_holistic,\
      mask_face, store_png,mp_drawing
import mediapipe as mp


def process_a_pic_by_blaze(pic_file, name, path=None):
    mp_image = mp.Image.create_from_file(pic_file)

    detect_results = detect_face_by_blaze_face(mp_image)
    masked_pic_mat = mask_face(mp_image, detect_results, 'blaze')
    store_png(masked_pic_mat, name, path)


def process_a_pic_by_face_solution(pic_file, name, path=None):
    mp_image = mp.Image.create_from_file(pic_file)

    detect_results = detect_face_by_face_solution(pic_file)
    masked_pic_mat = mask_face(mp_image, detect_results, 'face_solution')
    store_png(masked_pic_mat, name, path)

def process_a_pic_by_holistic(pic_file, name, path=None):
    mp_image = mp.Image.create_from_file(pic_file)

    detect_results = detect_face_by_holistic(pic_file)
    masked_pic_mat = mask_face(mp_image, detect_results, 'holistic')
    store_png(masked_pic_mat, name, path)