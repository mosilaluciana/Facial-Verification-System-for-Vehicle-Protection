from mtcnn.mtcnn import MTCNN
import cv2
import os

detector = MTCNN()

def crop_and_save_face(image_path, save_dir, img_size=105):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Nu pot citi {image_path}")
        return False

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_img)
    if len(results) == 0:
        print(f"[WARNING] Nu am găsit față în {image_path}")
        return False

    x, y, w, h = results[0]['box']
    x, y = max(x, 0), max(y, 0)
    face = rgb_img[y:y + h, x:x + w]
    if face.size == 0:
        print(f"[WARNING] Crop invalid pentru {image_path}")
        return False

    face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    return True
