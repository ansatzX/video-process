from videoprocess.utlis import detect_face, mask_face, store_png
import mediapipe as mp


def process_a_pic(pic_file, name, path=None):
    mp_image = mp.Image.create_from_file(pic_file)

    detct_results = detect_face(mp_image)
    masked_pic_mat = mask_face(mp_image, detct_results)
    store_png(masked_pic_mat, name, path)